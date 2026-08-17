package manager

import (
	"crypto/rand"
	"fmt"
	"strconv"
	"testing"
	"time"

	"github.com/inngest/inngest/pkg/cqrs"
	"github.com/inngest/inngest/pkg/enums"
	"github.com/inngest/inngest/pkg/tracing/meta"
	"github.com/inngest/inngest/pkg/tracing/metadata/extractors"
	"github.com/oklog/ulid/v2"
	"github.com/stretchr/testify/require"
)

func metadataSpanAttrs(kind, scope, values string) []byte {
	return fmt.Appendf(nil,
		`{"_inngest.metadata.kind":%q,"_inngest.metadata.scope":%q,"_inngest.metadata.op":"merge","_inngest.metadata.values":%s}`,
		kind, scope, strconv.Quote(values),
	)
}

func findAISummary(t *testing.T, span *cqrs.OtelSpan) []*cqrs.SpanMetadata {
	t.Helper()
	var out []*cqrs.SpanMetadata
	for _, md := range span.Metadata {
		if md.Kind == extractors.KindInngestAISummary {
			out = append(out, md)
		}
	}
	return out
}

func TestCQRSAISummaryMetadata(t *testing.T) {
	runAttr := []byte(`{"_inngest.dynamic.status":"Running"}`)
	stepAttrs := func(id string, attempt int) []byte {
		return fmt.Appendf(nil, `{"_inngest.step.id":%q,"_inngest.step.attempt":%d}`, id, attempt)
	}

	t.Run("sums counted scopes, excludes extended_trace, strips spoofed summaries", func(t *testing.T) {
		cm, cleanup := initCQRS(t)
		defer cleanup()

		runID := ulid.MustNew(ulid.Now(), rand.Reader).String()
		spans := []testSpanFields{
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: runAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: stepAttrs("a", 0)},
			{DynamicSpanID: "step2", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: stepAttrs("b", 0)},
			// Executor-reported usage at legacy step_attempt scope.
			{DynamicSpanID: "md1", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "step_attempt",
				`{"input_tokens":100,"output_tokens":20,"request_model":"gpt-4o","response_model":"gpt-4o-mini","estimated_cost":0.05}`,
			)},
			// Step-scoped usage with an explicit total that beats input+output.
			{DynamicSpanID: "md2", ParentSpanID: "step2", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "step",
				`{"input_tokens":10,"output_tokens":5,"total_tokens":40,"request_model":"claude-3-5"}`,
			)},
			// Extended-trace entries can duplicate step-level reporting and must
			// not be counted.
			{DynamicSpanID: "md3", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "extended_trace",
				`{"input_tokens":1000,"output_tokens":1000}`,
			)},
			// User-written run-scoped usage is additive.
			{DynamicSpanID: "md4", ParentSpanID: "root", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "run",
				`{"input_tokens":1,"output_tokens":2}`,
			)},
			// A stored summary must be stripped and recomputed, never trusted.
			{DynamicSpanID: "md5", ParentSpanID: "root", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai.summary", "run",
				`{"input_tokens":99999,"call_count":50}`,
			)},
		}
		for _, s := range spans {
			s.RunID = runID
			insertTestSpan(t, cm, s)
		}

		root, err := cm.GetSpansByRunID(t.Context(), ulid.MustParse(runID))
		require.NoError(t, err)
		require.NotNil(t, root)

		summaries := findAISummary(t, root)
		require.Len(t, summaries, 1)
		require.Equal(t, enums.MetadataScopeRun, summaries[0].Scope)

		sum, err := extractors.AISummaryFromValues(summaries[0].Values)
		require.NoError(t, err)
		require.Equal(t, int64(111), sum.InputTokens)
		require.Equal(t, int64(27), sum.OutputTokens)
		require.Equal(t, int64(163), sum.TotalTokens)
		require.NotNil(t, sum.EstimatedCost)
		require.InDelta(t, 0.05, *sum.EstimatedCost, 1e-9)
		// md1 reported both models, so the response model wins; md2 reported
		// only a request model, so that is used as the fallback.
		require.Equal(t, []string{"claude-3-5", "gpt-4o-mini"}, sum.Models)
		require.Equal(t, int64(3), sum.CallCount)
		require.False(t, sum.Partial)

		// The user's own run-scoped entry stays visible alongside the summary.
		userEntries := 0
		for _, md := range root.Metadata {
			if md.Kind == extractors.KindInngestAI {
				userEntries++
			}
		}
		require.Equal(t, 1, userEntries)
	})

	// Producers emit optional token counts as floats, which the step-level
	// AIMetadata struct types as *int64. Parsing must not be so strict that
	// such an entry's tokens are dropped from the summary.
	t.Run("counts entries whose optional fields are fractional", func(t *testing.T) {
		cm, cleanup := initCQRS(t)
		defer cleanup()

		runID := ulid.MustNew(ulid.Now(), rand.Reader).String()
		spans := []testSpanFields{
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: runAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: stepAttrs("a", 0)},
			{DynamicSpanID: "md1", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "step",
				`{"input_tokens":37,"output_tokens":6,"total_tokens":43,"cache_read_tokens":7.5,"request_model":"gpt-5.4-mini","estimated_cost":0.000055}`,
			)},
		}
		for _, s := range spans {
			s.RunID = runID
			insertTestSpan(t, cm, s)
		}

		root, err := cm.GetSpansByRunID(t.Context(), ulid.MustParse(runID))
		require.NoError(t, err)

		summaries := findAISummary(t, root)
		require.Len(t, summaries, 1)
		sum, err := extractors.AISummaryFromValues(summaries[0].Values)
		require.NoError(t, err)
		require.Equal(t, int64(1), sum.CallCount)
		require.Equal(t, int64(37), sum.InputTokens)
		require.Equal(t, int64(43), sum.TotalTokens)
		require.NotNil(t, sum.CacheReadTokens)
		require.Equal(t, int64(7), *sum.CacheReadTokens)
	})

	t.Run("sums granular tokens and providers, omitting unreported fields", func(t *testing.T) {
		cm, cleanup := initCQRS(t)
		defer cleanup()

		runID := ulid.MustNew(ulid.Now(), rand.Reader).String()
		spans := []testSpanFields{
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: runAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: stepAttrs("a", 0)},
			{DynamicSpanID: "step2", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: stepAttrs("b", 0)},
			{DynamicSpanID: "md1", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "step",
				`{"input_tokens":10,"output_tokens":2,"cache_read_tokens":7,"reasoning_tokens":4,"provider":"openai"}`,
			)},
			// No cache_read_tokens and no cache_creation_tokens anywhere but md1's
			// read count, so creation stays absent while read is still summed.
			{DynamicSpanID: "md2", ParentSpanID: "step2", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "step",
				`{"input_tokens":3,"output_tokens":1,"reasoning_tokens":6,"provider":"anthropic"}`,
			)},
		}
		for _, s := range spans {
			s.RunID = runID
			insertTestSpan(t, cm, s)
		}

		root, err := cm.GetSpansByRunID(t.Context(), ulid.MustParse(runID))
		require.NoError(t, err)

		summaries := findAISummary(t, root)
		require.Len(t, summaries, 1)
		sum, err := extractors.AISummaryFromValues(summaries[0].Values)
		require.NoError(t, err)

		require.NotNil(t, sum.CacheReadTokens)
		require.Equal(t, int64(7), *sum.CacheReadTokens)
		require.NotNil(t, sum.ReasoningTokens)
		require.Equal(t, int64(10), *sum.ReasoningTokens)
		require.Nil(t, sum.CacheCreationTokens)
		require.Equal(t, []string{"anthropic", "openai"}, sum.Providers)

		// Absent fields must not serialize as a misleading zero.
		require.NotContains(t, summaries[0].Values, "cache_creation_tokens")
	})

	t.Run("partial when the run invokes a child run", func(t *testing.T) {
		cm, cleanup := initCQRS(t)
		defer cleanup()

		runID := ulid.MustNew(ulid.Now(), rand.Reader).String()
		childRunID := ulid.MustNew(ulid.Now(), rand.Reader).String()
		invokeAttrs := fmt.Appendf(nil,
			`{"_inngest.step.id":"inv","_inngest.step.attempt":0,"_inngest.step.invoke.run.id":%q}`,
			childRunID,
		)

		spans := []testSpanFields{
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: runAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: invokeAttrs},
			{DynamicSpanID: "md1", ParentSpanID: "root", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "run", `{"input_tokens":5,"output_tokens":5}`,
			)},
		}
		for _, s := range spans {
			s.RunID = runID
			insertTestSpan(t, cm, s)
		}

		root, err := cm.GetSpansByRunID(t.Context(), ulid.MustParse(runID))
		require.NoError(t, err)

		summaries := findAISummary(t, root)
		require.Len(t, summaries, 1)
		sum, err := extractors.AISummaryFromValues(summaries[0].Values)
		require.NoError(t, err)
		require.True(t, sum.Partial)
		require.Equal(t, int64(10), sum.TotalTokens)
	})

	t.Run("no summary without AI usage or invokes", func(t *testing.T) {
		cm, cleanup := initCQRS(t)
		defer cleanup()

		runID := ulid.MustNew(ulid.Now(), rand.Reader).String()
		spans := []testSpanFields{
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: runAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: stepAttrs("a", 0)},
		}
		for _, s := range spans {
			s.RunID = runID
			insertTestSpan(t, cm, s)
		}

		root, err := cm.GetSpansByRunID(t.Context(), ulid.MustParse(runID))
		require.NoError(t, err)
		require.Empty(t, findAISummary(t, root))
	})
}

// Every inngest.ai emission under a given parent shares one dynamic_span_id,
// and the kind's merge opcode is last-write-wins — so rolling the group up
// into a single entry would discard all but the final emission's usage.
func TestCQRSAISummaryCountsEveryEmissionUnderOneParent(t *testing.T) {
	cm, cleanup := initCQRS(t)
	defer cleanup()

	runID := ulid.MustNew(ulid.Now(), rand.Reader)
	traceID := ulid.MustNew(ulid.Now(), rand.Reader).String()
	aiAttrs := func(in, out int) []byte {
		return metadataSpanAttrs("inngest.ai", "run", fmt.Sprintf(
			`{"input_tokens":%d,"output_tokens":%d}`, in, out,
		))
	}

	spans := []testSpanFields{
		{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: []byte(`{"_inngest.dynamic.status":"Completed"}`)},
		// Two separate emissions of the same kind under the same parent, as
		// CreateMetadataSpanFromValues produces them.
		{DynamicSpanID: "md-ai", ParentSpanID: "root", Name: meta.SpanNameMetadata, Attributes: aiAttrs(100, 10), StartTime: time.Now()},
		{DynamicSpanID: "md-ai", ParentSpanID: "root", Name: meta.SpanNameMetadata, Attributes: aiAttrs(200, 20), StartTime: time.Now().Add(time.Millisecond)},
	}
	for _, s := range spans {
		s.RunID = runID.String()
		s.TraceID = traceID
		insertTestSpan(t, cm, s)
	}

	root, err := cm.GetSpansByRunID(t.Context(), runID)
	require.NoError(t, err)

	summaries := findAISummary(t, root)
	require.Len(t, summaries, 1)
	tree, err := extractors.AISummaryFromValues(summaries[0].Values)
	require.NoError(t, err)
	require.Equal(t, int64(2), tree.CallCount, "the earlier emission must not be overwritten by the later one")
	require.Equal(t, int64(300), tree.InputTokens)
	require.Equal(t, int64(30), tree.OutputTokens)
	require.Equal(t, int64(330), tree.TotalTokens)

	usage, err := cm.GetRunsAIUsage(t.Context(), []ulid.ULID{runID})
	require.NoError(t, err)
	require.Equal(t, tree, usage[runID], "both aggregation paths must agree")
}

// Metadata spans reference their parent by ID with no existence check, so a
// dangling reference must still count toward the run's usage rather than
// silently vanishing from the tree-assembled summary.
func TestCQRSAISummaryCountsMetadataWithMissingParentSpan(t *testing.T) {
	cm, cleanup := initCQRS(t)
	defer cleanup()

	runID := ulid.MustNew(ulid.Now(), rand.Reader)
	spans := []testSpanFields{
		{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: []byte(`{"_inngest.dynamic.status":"Completed"}`)},
		{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: []byte(`{"_inngest.step.id":"a","_inngest.step.attempt":0}`)},
		{DynamicSpanID: "md1", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
			"inngest.ai", "step", `{"input_tokens":10,"output_tokens":5}`,
		)},
		// Parented on a step span that was never written.
		{DynamicSpanID: "md2", ParentSpanID: "does-not-exist", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
			"inngest.ai", "step", `{"input_tokens":100,"output_tokens":50}`,
		)},
	}
	for _, s := range spans {
		s.RunID = runID.String()
		insertTestSpan(t, cm, s)
	}

	root, err := cm.GetSpansByRunID(t.Context(), runID)
	require.NoError(t, err)

	summaries := findAISummary(t, root)
	require.Len(t, summaries, 1)
	tree, err := extractors.AISummaryFromValues(summaries[0].Values)
	require.NoError(t, err)
	require.Equal(t, int64(2), tree.CallCount)
	require.Equal(t, int64(110), tree.InputTokens)
	require.Equal(t, int64(55), tree.OutputTokens)

	usage, err := cm.GetRunsAIUsage(t.Context(), []ulid.ULID{runID})
	require.NoError(t, err)
	require.Equal(t, tree, usage[runID], "both aggregation paths must agree")
}

func TestCQRSGetRunsAIUsage(t *testing.T) {
	cm, cleanup := initCQRS(t)
	defer cleanup()

	completedAttr := []byte(`{"_inngest.dynamic.status":"Completed"}`)
	runningAttr := []byte(`{"_inngest.dynamic.status":"Running"}`)

	completedRun := ulid.MustNew(ulid.Now(), rand.Reader)
	runningRun := ulid.MustNew(ulid.Now(), rand.Reader)
	invokingRun := ulid.MustNew(ulid.Now(), rand.Reader)
	noUsageRun := ulid.MustNew(ulid.Now(), rand.Reader)
	missingRun := ulid.MustNew(ulid.Now(), rand.Reader)
	grandchildRun := ulid.MustNew(ulid.Now(), rand.Reader)

	fixtures := map[ulid.ULID][]testSpanFields{
		completedRun: {
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: completedAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: []byte(`{"_inngest.step.id":"a","_inngest.step.attempt":0}`)},
			{DynamicSpanID: "md1", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "step_attempt",
				`{"input_tokens":30,"output_tokens":12,"request_model":"gpt-4o","estimated_cost":0.01}`,
			)},
			// Extended-trace usage must be excluded here too.
			{DynamicSpanID: "md2", ParentSpanID: "step1", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "extended_trace", `{"input_tokens":500,"output_tokens":500}`,
			)},
		},
		runningRun: {
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: runningAttr},
			{DynamicSpanID: "md1", ParentSpanID: "root", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "run", `{"input_tokens":7,"output_tokens":3}`,
			)},
		},
		invokingRun: {
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: completedAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: fmt.Appendf(nil,
				`{"_inngest.step.id":"inv","_inngest.step.attempt":0,"_inngest.step.invoke.run.id":%q}`, grandchildRun.String())},
			{DynamicSpanID: "md1", ParentSpanID: "root", Name: meta.SpanNameMetadata, Attributes: metadataSpanAttrs(
				"inngest.ai", "run", `{"input_tokens":1,"output_tokens":1}`,
			)},
		},
		noUsageRun: {
			{DynamicSpanID: "root", Name: meta.SpanNameRun, Attributes: completedAttr},
			{DynamicSpanID: "step1", ParentSpanID: "root", Name: meta.SpanNameStep, Attributes: []byte(`{"_inngest.step.id":"a","_inngest.step.attempt":0}`)},
		},
	}
	for runID, spans := range fixtures {
		for _, s := range spans {
			s.RunID = runID.String()
			insertTestSpan(t, cm, s)
		}
	}

	usage, err := cm.GetRunsAIUsage(t.Context(), []ulid.ULID{completedRun, runningRun, invokingRun, noUsageRun, missingRun})
	require.NoError(t, err)

	completed, ok := usage[completedRun]
	require.True(t, ok)
	require.Equal(t, int64(30), completed.InputTokens)
	require.Equal(t, int64(12), completed.OutputTokens)
	require.Equal(t, int64(42), completed.TotalTokens)
	require.Equal(t, int64(1), completed.CallCount)
	require.Equal(t, []string{"gpt-4o"}, completed.Models)
	require.False(t, completed.Partial)

	// Still-executing runs are not partial here: the caller derives that from
	// the parent-side invoke span's status, so this never reads run spans.
	running, ok := usage[runningRun]
	require.True(t, ok)
	require.Equal(t, int64(10), running.TotalTokens)
	require.False(t, running.Partial)

	invoking, ok := usage[invokingRun]
	require.True(t, ok)
	require.Equal(t, int64(2), invoking.TotalTokens)
	require.True(t, invoking.Partial, "grandchild usage is beyond the caller's depth-1 fold")

	// Step spans alone are enough to be reported: a child with no usage of its
	// own could still invoke runs the caller can't see.
	noUsage, ok := usage[noUsageRun]
	require.True(t, ok)
	require.Equal(t, extractors.AISummaryMetadata{}, noUsage)

	_, ok = usage[missingRun]
	require.False(t, ok, "runs with no spans are omitted so callers can treat them as unreachable")
}
