| model                                | class   |   precision |   recall |    f1 |   support |
|:-------------------------------------|:--------|------------:|---------:|------:|----------:|
| ResNet18 (transfer learning)         | CNV     |       0.866 |    0.988 | 0.923 |       242 |
| ResNet18 (transfer learning)         | DME     |       0.98  |    1     | 0.99  |       242 |
| ResNet18 (transfer learning)         | DRUSEN  |       0.962 |    0.847 | 0.901 |       242 |
| ResNet18 (transfer learning)         | NORMAL  |       1     |    0.959 | 0.979 |       242 |
| YOLO classification (yolo11n-cls.pt) | CNV     |       0.983 |    0.983 | 0.983 |       242 |
| YOLO classification (yolo11n-cls.pt) | DME     |       0.984 |    0.992 | 0.988 |       242 |
| YOLO classification (yolo11n-cls.pt) | DRUSEN  |       0.992 |    0.983 | 0.988 |       242 |
| YOLO classification (yolo11n-cls.pt) | NORMAL  |       0.992 |    0.992 | 0.992 |       242 |
