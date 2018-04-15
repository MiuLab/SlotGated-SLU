# Slot-Gated Modeling for Joint Slot Filling and Intent Prediction

## Reference
Main paper to be cited

```
@inproceedings{goo2018slot,
  title={Slot-Gated Modeling for Joint Slot Filling and Intent Prediction},
    author={Chih-Wen Goo and Guang Gao and Yun-Kai Hsu and Chih-Li Huo and Tsung-Chieh Chen and Keng-Wei Hsu and Yun-Nung Chen},
    booktitle={Proceedings of The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies},
    year={2018}
}
```

## Data
split your data into text file(seq.in), slot file(seq.out) and intent file(label) <br>
following is default setting used by train.py <br>
./data/ <br>
&emsp;--train/ <br>
&emsp;&emsp;--seq.in <br>
&emsp;&emsp;--seq.out <br>
&emsp;&emsp;--label <br>
&emsp;--test/ <br>
&emsp;&emsp;--seq.in <br>
&emsp;&emsp;--seq.out <br>
&emsp;&emsp;--label <br>
&emsp;--valid/ <br>
&emsp;&emsp;--seq.in <br>
&emsp;&emsp;--seq.out <br>
&emsp;&emsp;--label

## Requirements
tensorflow 1.4 <br>
python 3.5

## Usage
some sample usage <br>
* run with 32 units and no patience for early stop <br>
&emsp;python3 train.py --num_units=32 --patience=0

* disable early stop and use intent attention version <br>
&emsp;python3 train.py --no_early_stop --model_type=intent_only

* use "python3 train.py -h" for all avaliable parameter settings
