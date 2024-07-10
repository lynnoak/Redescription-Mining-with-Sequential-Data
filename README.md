# RMSP: Redescription Mining Extended with Sequential Pattern Mining

This project focuses on extending redescription mining with sequential pattern mining. The approach involves two primary views: a sequential view \(V_s\) and a non-sequential view \(V_a\).

## Methodology

### Steps:

1. **Sequential Pattern Mining on \(V_s\)**:
    - Apply sequential pattern mining on \(V_s\) to find candidate subsequence patterns.

2. **Initial Label Selection**:
    - Select the initial labels from \(V_a\) and \(V_s\).

3. **Processing Each Initial Label**:
    - For each initial label:
        1. **Decision Tree Growth**:
            - Grow a decision tree over \(V_a\) based on the target label.
            - Extract a query \(q_a\) from the decision tree.
        2. **Query Matching**:
            - Search the mined subsequence patterns.
            - Find a matching query \(q_s\) with \(q_a\) to complete the pair. 
            - Grow a new decision tree over \(V_a\) based on \(supp(q_s)\).
            - Find a new query \(q'_a\).
            - Find a new matching query \(q_s\) from subsequence patterns with \(q'_a\) to complete the pair.
        4. **Evaluation and Update**:
            - If the new query pair \((q'_a, q_s)\) satisfies the given constraints and is better than the current best one, replace the best one and update the target label with \(q_s\).
            - Otherwise, update the target label with some random noise.
        5. **Alternation**:
            - Continue alternating in this way until no further improvement can be achieved.

4. **Repeat**:
    - Repeat the above loop to get a set of mined query pairs as the redescription candidates.

