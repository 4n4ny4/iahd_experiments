# Retrieval Head Overlap Analysis

Model: Llama-3-8B-Instruct (32L x 32H = 1024 heads)

## Per-Task Summary

| Task                         | Attempts | ROUGE OK | Value Match | #Heads | %Heads |
|------------------------------|----------|----------|-------------|--------|--------|
| ceo_lastname                 |       98 |        0 |          55 |      0 |   0.0% |
| employees_count_full_time    |       49 |       22 |           8 |     11 |   1.1% |
| employees_count_total        |      100 |       42 |           8 |     19 |   1.9% |
| headquarters_city            |      100 |        0 |          66 |      0 |   0.0% |
| headquarters_state           |      100 |        9 |          57 |     24 |   2.3% |
| holder_record_amount         |       98 |       39 |          20 |     22 |   2.1% |
| incorporation_state          |       98 |        6 |          93 |     21 |   2.1% |
| incorporation_year           |      100 |       16 |          71 |     25 |   2.4% |
| registrant_name              |       99 |        4 |          96 |     13 |   1.3% |

## Overlap Statistics

- Union of all retrieval heads: **34** (3.3%)
- Intersection of all retrieval heads: **0** (0.0%)

## Jaccard Overlap Matrix

|  | ceo_lastname | employees_co | employees_co | headquarters | headquarters | holder_recor | incorporatio | incorporatio | registrant_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ceo_lastname | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| employees_co | 0.00 | 1.00 | 0.58 | 0.00 | 0.46 | 0.50 | 0.45 | 0.44 | 0.20 |
| employees_co | 0.00 | 0.58 | 1.00 | 0.00 | 0.72 | 0.86 | 0.60 | 0.63 | 0.23 |
| headquarters | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| headquarters | 0.00 | 0.46 | 0.72 | 0.00 | 1.00 | 0.77 | 0.73 | 0.75 | 0.28 |
| holder_recor | 0.00 | 0.50 | 0.86 | 0.00 | 0.77 | 1.00 | 0.72 | 0.68 | 0.30 |
| incorporatio | 0.00 | 0.45 | 0.60 | 0.00 | 0.73 | 0.72 | 1.00 | 0.77 | 0.31 |
| incorporatio | 0.00 | 0.44 | 0.63 | 0.00 | 0.75 | 0.68 | 0.77 | 1.00 | 0.27 |
| registrant_n | 0.00 | 0.20 | 0.23 | 0.00 | 0.28 | 0.30 | 0.31 | 0.27 | 1.00 |

## Top Pairwise Overlaps

- employees_count_total x holder_record_amount: Jaccard = 0.864
- headquarters_state x holder_record_amount: Jaccard = 0.769
- incorporation_state x incorporation_year: Jaccard = 0.769
- headquarters_state x incorporation_year: Jaccard = 0.750
- headquarters_state x incorporation_state: Jaccard = 0.731
- employees_count_total x headquarters_state: Jaccard = 0.720
- holder_record_amount x incorporation_state: Jaccard = 0.720
- holder_record_amount x incorporation_year: Jaccard = 0.679
- employees_count_total x incorporation_year: Jaccard = 0.630
- employees_count_total x incorporation_state: Jaccard = 0.600

## Key Research Findings

1. **Average pairwise Jaccard overlap**: 0.312
2. **Retrieval heads are moderately shared** across the 9 tested tasks.
3. **Head sparsity**: Tasks use between 0 and 25 retrieval heads (0.0%-2.4% of all heads).
4. **Core retrieval heads**: 0 heads are retrieval heads for *every* task tested.