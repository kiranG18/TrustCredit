# Notebook Modernization Fixes

The following table lists the modernization fixes applied to the Jupyter Notebooks to update deprecated imports to the current package layout:

| Notebook | Deprecated Code | Modernized Code | Description |
| :--- | :--- | :--- | :--- |
| **`01_baseline_training.ipynb`** | `from trustcredit.training import optimize_model_fast, create_pipeline` | `from trustcredit.features.pipeline import create_pipeline`<br>`from trustcredit.models.tuning import optimize_model_fast` | Updated imports for pipeline and tuning modules. |
| **`01_baseline_training.ipynb`** | `from trustcredit.evaluate import get_metrics` | `from trustcredit.evaluation import get_metrics` | Fixed evaluation module import. |
| **`02_fairness_methods.ipynb`** | `from trustcredit import training` | `from trustcredit.features.pipeline import create_pipeline` | Updated training module import to features pipeline. |
| **`02_fairness_methods.ipynb`** | `training.create_pipeline` | `create_pipeline` | Directly call the features pipeline creator. |
| **`02_fairness_methods.ipynb`** | `from trustcredit.evaluate import get_fairness_metrics` | `from trustcredit.evaluation import get_fairness_metrics` | Fixed evaluation module import. |
| **`02_fairness_methods.ipynb`** | `from trustcredit.fairness_models import Reweighing, FairGBM, ThresholdOpt` | `from trustcredit.fairness import Reweighing, FairGBM, ThresholdOpt` | Fixed fairness module import. |
| **`03_reject_inference.ipynb`** | `from trustcredit import training` | `from trustcredit.features.pipeline import create_pipeline` | Updated training module import. |
| **`03_reject_inference.ipynb`** | `training.create_pipeline` | `create_pipeline` | Directly call the features pipeline creator. |
| **`03_reject_inference.ipynb`** | `from trustcredit.evaluate import get_reject_inference_metrics` | `from trustcredit.evaluation import get_reject_inference_metrics` | Fixed evaluation module import. |
| **`04_explainability.ipynb`** | `from trustcredit.training import optimize_model_fast, create_pipeline` | `from trustcredit.features.pipeline import create_pipeline`<br>`from trustcredit.models.tuning import optimize_model_fast` | Updated pipeline and tuning imports. |
| **`04_explainability.ipynb`** | `from trustcredit.evaluate import get_metrics` | `from trustcredit.evaluation import get_metrics` | Fixed evaluation module import. |

### Verification
All notebooks have been validated and can now be imported and run in a clean environment containing the core `trustcredit` package dependencies.
