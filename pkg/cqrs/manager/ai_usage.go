package manager

import (
	"context"
	"fmt"
	"slices"

	"github.com/inngest/inngest/pkg/cqrs"
	"github.com/inngest/inngest/pkg/logger"
	"github.com/inngest/inngest/pkg/tracing/meta"
	"github.com/inngest/inngest/pkg/tracing/metadata/extractors"
	"github.com/oklog/ulid/v2"
	"golang.org/x/sync/errgroup"
)

// addCountedAIMetadata folds every counted inngest.ai entry in mds into the
// builder. It is the single definition of "what counts", shared by the span
// tree aggregation in cqrs.go and by GetRunsAIUsage, so the two can never drift.
func addCountedAIMetadata(ctx context.Context, builder *extractors.AISummaryBuilder, mds []*cqrs.SpanMetadata, logAttrs ...any) {
	for _, md := range mds {
		if md.Kind != extractors.KindInngestAI || !extractors.AIUsageScopeCounted(md.Scope) {
			continue
		}
		if err := builder.AddCall(md.Values); err != nil {
			logger.StdlibLogger(ctx).Warn(
				"skipping malformed inngest.ai metadata entry",
				append(slices.Clone(logAttrs), "error", err)...,
			)
		}
	}
}

// GetRunsAIUsage sums each run's counted inngest.ai metadata into a run-level
// AI usage summary, reading only metadata and step spans rather than
// assembling the full span tree. It exists so the GraphQL loader layer can
// fold step.invoke child run usage into a parent run's inngest.ai.summary;
// GetSpansByRunID must never call it, since that primitive also serves the
// rerun path.
//
// A run's summary is marked partial when it invokes runs of its own, since
// that usage is beyond the caller's depth-1 fold. Whether the run is still
// executing is not considered: the caller derives that from the parent-side
// invoke span. Runs with neither metadata nor step spans are omitted so
// callers can treat them as unreachable.
func (w wrapper) GetRunsAIUsage(ctx context.Context, runIDs []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
	if len(runIDs) == 0 {
		return map[ulid.ULID]extractors.AISummaryMetadata{}, nil
	}

	var metadataSpans, stepSpans map[ulid.ULID][]*cqrs.OtelSpan
	eg, egctx := errgroup.WithContext(ctx)
	eg.Go(func() error {
		var err error
		metadataSpans, err = w.GetSpansByRunIDsAndName(egctx, runIDs, meta.SpanNameMetadata)
		return err
	})
	eg.Go(func() error {
		var err error
		stepSpans, err = w.GetSpansByRunIDsAndName(egctx, runIDs, meta.SpanNameStep)
		return err
	})
	if err := eg.Wait(); err != nil {
		return nil, fmt.Errorf("error loading spans for AI usage: %w", err)
	}

	out := make(map[ulid.ULID]extractors.AISummaryMetadata, len(runIDs))
	for _, runID := range runIDs {
		// A run with step spans but no usage of its own must still be reported:
		// omitting it would let the caller read a complete summary while the
		// run's own invokes hide grandchild usage.
		if len(metadataSpans[runID]) == 0 && len(stepSpans[runID]) == 0 {
			continue
		}

		builder := extractors.NewAISummaryBuilder()
		for _, span := range metadataSpans[runID] {
			addCountedAIMetadata(ctx, builder, span.Metadata, "run_id", runID.String())
		}

		for _, span := range stepSpans[runID] {
			if span.IsInvokeStep() {
				builder.MarkPartial()
				break
			}
		}

		out[runID] = builder.Summary()
	}

	return out, nil
}
