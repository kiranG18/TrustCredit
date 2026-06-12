# Counterfactual Explanations (Actionable Recourse)

The table below shows how the rejected applicant can modify their application attributes (e.g., lower credit amount or loan duration) to change the model's outcome from **Default (Rejected)** to **Approved**.

|              | Original    | CF 0            | CF 1             | CF 2             |
|:-------------|:------------|:----------------|:-----------------|:-----------------|
| Duration     | 38 (0.0)    | 7 (--30.9)      | 8 (--30.5)       | 7 (--31.0)       |
| CreditAmount | 17707 (0.0) | 16743 (--963.9) | 16497 (--1210.2) | 16530 (--1176.9) |
| Age          | 34 (0.0)    | 32 (--2.0)      | 30 (--4.1)       | 27 (--6.6)       |
