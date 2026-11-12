# DeSTSeg

DeSTSeg trains a denoising student against a frozen ImageNet teacher and then trains a segmentation network on synthetic anomalies. See [the paper](https://openaccess.thecvf.com/content/CVPR2023/html/Zhang_DeSTSeg_Segmentation_Guided_Denoising_Student-Teacher_for_Anomaly_Detection_CVPR_2023_paper.html) and [the reference implementation](https://github.com/apple-aiml-research/ml-destseg).

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Destseg

model = Destseg(anomaly_source_path="datasets/dtd/images")
engine = Engine()
engine.fit(model=model, datamodule=MVTecAD())
```

The texture directory is optional; without it, synthetic noise supplies the anomalies. The default schedule trains the student for 1,000 steps and the segmentation network for 4,000 steps. The teacher downloads ImageNet weights on first use.

Training expects inputs normalized with ImageNet mean and standard deviation, as supplied by the default pre-processor. Custom pre-processors must apply the same normalization.
