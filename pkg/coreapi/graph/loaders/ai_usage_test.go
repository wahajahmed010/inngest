package loader

import (
	"context"
	"crypto/rand"
	"fmt"
	"testing"

	"github.com/inngest/inngest/pkg/cqrs"
	"github.com/inngest/inngest/pkg/enums"
	"github.com/inngest/inngest/pkg/tracing/meta"
	"github.com/inngest/inngest/pkg/tracing/metadata/extractors"
	"github.com/oklog/ulid/v2"
	"github.com/stretchr/testify/require"
)

func summarySpan(t *testing.T, sum extractors.AISummaryMetadata, children ...*cqrs.OtelSpan) *cqrs.OtelSpan {
	t.Helper()
	values, err := sum.Serialize()
	require.NoError(t, err)
	return &cqrs.OtelSpan{
		Metadata: []*cqrs.SpanMetadata{{
			Scope:  enums.MetadataScopeRun,
			Kind:   extractors.KindInngestAISummary,
			Values: values,
		}},
		Children: children,
	}
}

func invokeSpan(childRunID ulid.ULID, status enums.StepStatus) *cqrs.OtelSpan {
	return &cqrs.OtelSpan{
		Status:     status,
		Attributes: &meta.ExtractedValues{StepInvokeRunID: &childRunID},
	}
}

func parseSummary(t *testing.T, root *cqrs.OtelSpan) extractors.AISummaryMetadata {
	t.Helper()
	for _, md := range root.Metadata {
		if md.Kind == extractors.KindInngestAISummary {
			sum, err := extractors.AISummaryFromValues(md.Values)
			require.NoError(t, err)
			return sum
		}
	}
	t.Fatal("no inngest.ai.summary entry on root span")
	return extractors.AISummaryMetadata{}
}

func TestFoldChildRunAIUsage(t *testing.T) {
	ctx := context.Background()
	childID := ulid.MustNew(ulid.Now(), rand.Reader)
	cost := 0.02

	childReasoning := int64(12)

	inTree := extractors.AISummaryMetadata{
		InputTokens:  10,
		OutputTokens: 5,
		TotalTokens:  15,
		Models:       []string{"claude-3-5"},
		Providers:    []string{"anthropic"},
		CallCount:    1,
		Partial:      true,
	}

	t.Run("folds a resolved child's usage into the root summary", func(t *testing.T) {
		root := summarySpan(t, inTree, invokeSpan(childID, enums.StepStatusCompleted))

		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			require.Equal(t, []ulid.ULID{childID}, ids)
			return map[ulid.ULID]extractors.AISummaryMetadata{
				childID: {
					InputTokens:     90,
					OutputTokens:    45,
					TotalTokens:     135,
					EstimatedCost:   &cost,
					ReasoningTokens: &childReasoning,
					Models:          []string{"gpt-4o"},
					Providers:       []string{"openai"},
					CallCount:       2,
				},
			}, nil
		})

		sum := parseSummary(t, root)
		require.Equal(t, int64(100), sum.InputTokens)
		require.Equal(t, int64(50), sum.OutputTokens)
		require.Equal(t, int64(150), sum.TotalTokens)
		require.NotNil(t, sum.EstimatedCost)
		require.InDelta(t, 0.02, *sum.EstimatedCost, 1e-9)
		require.Equal(t, []string{"claude-3-5", "gpt-4o"}, sum.Models)
		require.Equal(t, []string{"anthropic", "openai"}, sum.Providers)
		// Only the child reported reasoning tokens, so its count carries through.
		require.NotNil(t, sum.ReasoningTokens)
		require.Equal(t, int64(12), *sum.ReasoningTokens)
		require.Nil(t, sum.CacheReadTokens)
		require.Equal(t, int64(3), sum.CallCount)
		require.False(t, sum.Partial, "a resolved child with a complete summary clears partial")
	})

	t.Run("stays partial when a child is unreachable", func(t *testing.T) {
		root := summarySpan(t, inTree, invokeSpan(childID, enums.StepStatusCompleted))

		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			return map[ulid.ULID]extractors.AISummaryMetadata{}, nil
		})

		sum := parseSummary(t, root)
		require.Equal(t, int64(15), sum.TotalTokens)
		require.True(t, sum.Partial)
	})

	t.Run("stays partial when a child reports partial usage", func(t *testing.T) {
		root := summarySpan(t, inTree, invokeSpan(childID, enums.StepStatusCompleted))

		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			return map[ulid.ULID]extractors.AISummaryMetadata{
				childID: {TotalTokens: 100, CallCount: 1, Partial: true},
			}, nil
		})

		sum := parseSummary(t, root)
		require.Equal(t, int64(115), sum.TotalTokens)
		require.True(t, sum.Partial)
	})

	t.Run("stays partial while an invoke is unresolved", func(t *testing.T) {
		root := summarySpan(t, inTree, invokeSpan(childID, enums.StepStatusInvoking))

		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			return map[ulid.ULID]extractors.AISummaryMetadata{
				childID: {TotalTokens: 100, CallCount: 1},
			}, nil
		})

		sum := parseSummary(t, root)
		require.Equal(t, int64(115), sum.TotalTokens)
		require.True(t, sum.Partial)
	})

	t.Run("keeps the in-tree summary when the fetch fails", func(t *testing.T) {
		root := summarySpan(t, inTree, invokeSpan(childID, enums.StepStatusCompleted))

		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			return nil, fmt.Errorf("boom")
		})

		sum := parseSummary(t, root)
		require.Equal(t, int64(15), sum.TotalTokens)
		require.True(t, sum.Partial)
	})

	noUsage := func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
		out := map[ulid.ULID]extractors.AISummaryMetadata{}
		for _, id := range ids {
			out[id] = extractors.AISummaryMetadata{}
		}
		return out, nil
	}

	hasSummary := func(root *cqrs.OtelSpan) bool {
		for _, md := range root.Metadata {
			if md.Kind == extractors.KindInngestAISummary {
				return true
			}
		}
		return false
	}

	// The invoke step span is updated in place when its pause resumes, so its
	// terminal statuses are the parent-side proof that no further child usage
	// can arrive.
	for _, status := range []enums.StepStatus{
		enums.StepStatusCompleted,
		enums.StepStatusFailed,
		enums.StepStatusErrored,
		enums.StepStatusTimedOut,
		enums.StepStatusCancelled,
		enums.StepStatusSkipped,
	} {
		t.Run("drops an empty placeholder once the invoke is "+status.String(), func(t *testing.T) {
			root := summarySpan(t, extractors.AISummaryMetadata{Partial: true}, invokeSpan(childID, status))

			foldChildRunAIUsage(ctx, root, noUsage)

			require.False(t, hasSummary(root))
		})
	}

	for _, status := range []enums.StepStatus{
		enums.StepStatusInvoking,
		enums.StepStatusRunning,
	} {
		t.Run("keeps an empty placeholder while the invoke is "+status.String(), func(t *testing.T) {
			root := summarySpan(t, extractors.AISummaryMetadata{Partial: true}, invokeSpan(childID, status))

			foldChildRunAIUsage(ctx, root, noUsage)

			require.True(t, hasSummary(root))
			require.True(t, parseSummary(t, root).Partial)
		})
	}

	t.Run("no-op without invoked children", func(t *testing.T) {
		root := summarySpan(t, extractors.AISummaryMetadata{TotalTokens: 15, CallCount: 1})

		called := false
		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			called = true
			return nil, nil
		})
		require.False(t, called)

		sum := parseSummary(t, root)
		require.Equal(t, int64(15), sum.TotalTokens)
	})

	t.Run("keeps an empty placeholder while a sibling invoke has no stamped run ID", func(t *testing.T) {
		op := enums.OpcodeInvokeFunction
		pendingInvoke := &cqrs.OtelSpan{Attributes: &meta.ExtractedValues{StepOp: &op}}
		root := summarySpan(
			t,
			extractors.AISummaryMetadata{Partial: true},
			invokeSpan(childID, enums.StepStatusCompleted),
			pendingInvoke,
		)

		foldChildRunAIUsage(ctx, root, noUsage)

		require.True(t, hasSummary(root), "an invoke step whose child run isn't stamped yet is still outstanding")
	})

	t.Run("collects invoked run IDs from nested spans", func(t *testing.T) {
		otherID := ulid.MustNew(ulid.Now(), rand.Reader)
		nested := &cqrs.OtelSpan{
			Status:   enums.StepStatusCompleted,
			Children: []*cqrs.OtelSpan{invokeSpan(otherID, enums.StepStatusCompleted)},
		}
		root := summarySpan(t, inTree, invokeSpan(childID, enums.StepStatusCompleted), nested)

		var got []ulid.ULID
		foldChildRunAIUsage(ctx, root, func(ctx context.Context, ids []ulid.ULID) (map[ulid.ULID]extractors.AISummaryMetadata, error) {
			got = ids
			return map[ulid.ULID]extractors.AISummaryMetadata{
				childID: {TotalTokens: 100, CallCount: 1},
				otherID: {TotalTokens: 50, CallCount: 1},
			}, nil
		})

		require.ElementsMatch(t, []ulid.ULID{childID, otherID}, got)
		sum := parseSummary(t, root)
		require.Equal(t, int64(165), sum.TotalTokens)
		require.False(t, sum.Partial)
	})
}
