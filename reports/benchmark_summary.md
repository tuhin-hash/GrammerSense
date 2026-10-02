# GEC Benchmark Evaluation Report

**Generated**: `2026-10-02T06:02:55.464493+00:00`

## 1. Reproducibility Configuration
| Parameter | Value |
|---|---|
| **Model Checkpoint** | `vennify/t5-base-grammar-correction` |
| **Execution Device** | `cpu` |
| **Random Seed** | `42` |
| **Beam Search Size** | `4` |
| **Dataset Path** | `data\processed\test.jsonl` |
| **Evaluated Test Samples** | `117` |

## 2. Quantitative Benchmark Results
| Metric | Score | Description |
|---|---|---|
| **Corpus GLEU** | **52.2%** | Google GLEU (JFLEG standard, rewards correct edits) |
| **Baseline GLEU** | 43.1% | Score if zero edits are performed |
| **GLEU Improvement** | **+9.1%** | Net grammatical fluency gain |
| **Corpus BLEU** | 58.99% | 4-gram precision with brevity penalty |
| **Token Precision** | 21.02% | Accuracy of suggested edits |
| **Token Recall** | 30.0% | Proportion of gold errors corrected |
| **Token $F_{0.5}$ Score** | **22.36%** | Standard GEC metric (Precision weighted 2x over Recall) |
| **Exact Match (EM)** | 5.98% | Full-sentence exact match with reference |

## 3. Edit Classification Breakdown
- **True Positives (Correct Edits)**: `66`
- **False Positives (Spurious Edits)**: `248`
- **False Negatives (Missed Edits)**: `154`

## 4. Sample Test Predictions
| # | Source Sentence | Model Hypothesis | Gold Reference | GLEU |
|---|---|---|---|---|
| 1 | Natural language processing is a branch of artificial intelligence. | **Natural language processing is a branch of artificial intelligence.** | Natural language processing is a branch of artificial intelligence. | 1.0 |
| 2 | This will help to creat brotherhood realtionship . | **This will help to create brotherhood relationships.** | This will help to create brotherhood . | 0.8091 |
| 3 | They have to contact variety fields to find out what their interests are . | **They have to contact different fields to find out what their interests are.** | They have to contact a variety of fields to find out what their interests are . | 0.5322 |
| 4 | The first is our life condition . | **The first is our life condition.** | The first is our life 's conditions . | 0.5444 |
| 5 | Young people are more easily to adapt new technologies than old people . | **Young people are more easily able to adapt new technologies than old people.** | Young people are more easily able to adapt to new technologies than old people . | 0.5345 |
| 6 | In this way a person always feels to be original and `` modern '' , not `` ancient '' like people that surroand him or her . | **In this way, a person always feels original and  modern '' , not  ancient '' like the people that surround him or her.** | In this way , a person always feels original and `` modern , '' not `` ancient '' like the people that surround him or her . | 0.4445 |
| 7 | bacause I have broad knowledge . | **Because I have broad knowledge.** | Because I have broad knowledge . | 0.5475 |
| 8 | Maybe for the ecologistics in twenty years there will be fewer cars in use than there are today not for the citizens that everyday must go to the office . | **Maybe for the ecologists, in twenty years there will be fewer cars in use than there are today, but not for the citizens that must go to the office everyday.** | Maybe , for the ecologists , in twenty years there will be fewer cars in use than there are today , but not for the citizens that must go to the office everyday . | 0.6524 |
| 9 | They can travel more freely than older people do . | **They can travel more freely than older people do.** | They can travel more freely than older people do . | 0.7726 |
| 10 | It is very important for theme to understand ideas and concepts than it is for them to learn facts . | **It is very important for them to understand ideas and concepts than it is for them to learn facts.** | It is more important for them to understand ideas and concepts than it is for them to learn facts . | 0.6524 |
