# Review Checklist

## Architectural Questions

Ask these first when reviewing an FP-informed change:

1. Can the computational logic be isolated into a pure helper?
2. Is an expected failure still hidden behind `None`, `null`-like semantics, or unchecked exceptions?
3. Can domain state be made more explicit with a named type or enum?
4. Does the change mutate shared state without a strong reason?
5. Can the code become simpler through value-returning expressions and smaller steps?
6. If mutation remains, is it local, isolated, and justified?
7. Did the PR add an abstraction level that is smarter than the problem requires?

## Coverage Questions

Ask these next:

1. Is there still a test through the preserved public boundary?
2. Is the new pure collaborator tested directly?
3. Did the change add invariant coverage where the behavior should hold generally?
4. Are failure paths covered as behavior, not just line execution?
5. Are tests coupled to implementation details instead of public outcomes?

## Review Output Style

When asked for a review:

- present findings first
- order by severity
- cite the missing or risky behavior
- mention missing tests explicitly
- keep summaries short