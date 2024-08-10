# 가상환경
## 가상환경 설정
```bash
python3 -m venv .venv
```
## 가상환경 활성화
```bash
source ./myenv/bin/activate
```
## 가상환경 비활성화
```bash
deactivate
```
__python 3.8.10 버전으로 작성되었습니다.__
# pytorch lightning
## LightningModule
Lightning Module은 6가지로 구성됩니다.

* Computations (init).
* Train loop (training_step)
    * 단일 batch에서 loss를 반환하고, train loop로 자동 반복된다. 즉 학습 loop의 body를 나타낸다.
* Validation loop (validation_step)
* Test loop (test_step)
* Prediction loop (predict_step)
* Optimizers (configure_optimizers)
    * optimizer와 scheduler 구현 
[lightning 한국어 설명](https://wikidocs.net/157586)
## save_hyperparameters()
lightingmodule의 method이다.
* `__init__`에 있는 매개변수 (hyperparameter) 값을 `self.hparams`에 저장한다.
### 사용법
```python
class LitMNIST(L.LightningModule):
    def __init__(self, layer_1_dim=128, learning_rate=1e-2):
        super().__init__()
        # call this to save (layer_1_dim=128, learning_rate=1e-4) to the checkpoint
        self.save_hyperparameters()

        # equivalent
        self.save_hyperparameters("layer_1_dim", "learning_rate")

        # Now possible to access layer_1_dim from hparams
        self.hparams.layer_1_dim
```
츨처: [pytorch lighting api - save_hyperpameters](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html)

## LightningDataModule
* prepare_data
* setup
* train_dataloader
* val_dataloader
* test_dataloader

## ModelCheckPoint
`monitor` key에 대한 값을 checkpoint로 저장하는 모듈
* __monitor__: checkpoint를 저장하는 기준, mode 매개변수랑 관련이 있다.
* __mode__: {'min, max'} 현재 모니터링 된 값이 __mode__ 기준으로 변경하면 checkpoint를 기록한다. 보통 _monitor_ 값이 `val_acc` 이면 mode는 `max`, _monitor_ 값이 `val_loss` 이면 mode는 `min`으로 자동으로 적용한다.
* __dirpath__: checkpoint를 저장하는 위치

## EarlyStopping
모델 성능이 향상되었을 때, 일찍 학습을 종료하는 기준을 설정하는 모듈
* __monitor__: 감시하는 값 
* __patience__: 성능 향상의 변화가 없을 때 종료하지 않고 기다리는 최대 횟수 -> 정의하지 않으면 매 epoch마다 확인한다.

## Trainer
model train package
* __fast_dev_run__: 전체 데이터로 테스트하는 것이 아니라 소수의 데이터로 테스트하는 디버깅 목적 flag -> n(int)면 n개의 batch 실험
    * 해당 option을 `True`로 설정하면, `tuner`, `checkpoint callbacks`, `ealry stopping callbacks`, `logger`, `logger callbacks` 가 비활성화됨
    * 1 epoch에 대해서만 실행됨


# huggingface API
## datasets.load_data()
여러 데이터를 불러올 수 있음
### datasets.load_data("glue", "cola")
__cola__
* __Size of downloaded dataset files__: 0.38 MB
* __Size of the generated dataset__: 0.61 MB
* __Total amount of disk used__: 0.99 MB  
An example of 'train' looks as follows.
```
{
  "sentence": "Our friends won't buy this analysis, let alone the next one we propose.",
  "label": 1,
  "id": 0
}
```

### set_format()
__기본 데이터__
```
Dataset({
    features: ['attention_mask', 'idx', 'input_ids', 'label', 'sentence', 'token_type_ids'],
    num_rows: 8551
})
```
set format의 column을 쓰면, 해당 column에 있는 값만 추출된다.  
아무것도 안쓰면 전체를 return

```
train_data.set_format(
                type="torch", columns=["input_ids", "attention_mask", "label"]
)
```

## BertModel
### last_hidden_state[:, 0, :]
미리 학습한 bert 모델의 맨 마지막 hidden state의 cls token을 가져오는 작업
* last_hidden_state의 shape는 (batch size, input token size, vector dim)
* 왜 cls를 사용하는가? 다른 모든 token의 정보를 모두 담은 token이기 때문 -> __[CLS] a good representation for sentence-level classification__
* cls는 항상 자리가 고정되어 있기 때문에 정보를 쉽게 얻을 수 있다.-> 그렇기 때문에 사용하고 문맥의 모든 정보를 얻을 수 있음.

# GLUE banchmark
__GLUE__ (General Language Understanding Evaluation)
| 인간의 언어 능력을 인공지능이 얼마나 따라왔는지 정량적 성능 지표로 만든 것
## GLUE Tasks
논문에서 총 9개의 영어 문장 이해 task를 제공하며, _일반화 가능한 자연어 이해 시스템을 발전 시키는 것이 목표_
### Single-Sentence Tasks
* __CoLA__  
    * The Corpus of Linguistic Acceptability
    * 언어 학적으로 수용 가능한 문장인지 판별  
    * 기본적으로 이진 분류 문제
* __SST-2__
    * The Stanford Sentiment Treebank
    * 영화 리뷰와 그에 따른 pos/neg label이 있는 데이터
    * 문장이 주어지면 감정 분석을 통해서 문장의 긍정/부정 여부를 이진분류
### Similarity and Paraphrase Tasks
* __MRPC__
    * The Microsoft Research Paraphrase Corpus
    * 문장 쌍과 그에 대한 라벨로 이루어져 있다.
    * 문장 쌍이 의미론적으로 동일한지 판별해서 이진분류를 수행
* __QQP__
    * The Quora Question Pairs dataset
    * 질문 쌍과 그에 대한 라벨로 이루어져 있다.
    * 두개의 질문이 의미론적으로 같은지 여부를 판별한다.
* __STS-B__
    * The Semantic Textual Similarity Benchmark
    * 뉴스 헤드라인, 이미지 캡션, 자연어 추론 데이터에서 가져온 문장 쌍
    * 인간이 문장 쌍에 대한 유사도를 1~5로 라벨링 해놓은 데이터이고, 이를 예측
### Inference Tasks
* __MNLI__
    * The Multi-Genre Natural Language Corpus
    * 전체 (premise)와 가설 (hypothesis) 쌍으로 이루어져 있고 이에 대한 언어적 함의 (textual entailment) 라벨링이 되어있음
    * 이 문장 관계를 세가지로 예측 한다.
        * entailment(가설이 전제의 함의를 담고 있음)
        * contradiction(가설과 전제가 모순)
        * neutral(둘다 아닐 때)
* __QNLI__
    * The Stanford Question Answering Dataset
    * 문단-질문 쌍으로 이루어져 있고, 문단 중 한 문장이 질문에 대한 답을 담고 있다.
    * 문단의 원래 문장이 질문에 대한 답을 담고 있는지 분류한다.
    * _단순히 겹치는 어휘쌍이 많은 문장 쌍_ 을 정답으로 추론하는 _간단한 추론_ 을 모델이 하지 못하게 해서 모델 성능을 잘 측정할 수 있게 한다.
* __RTE__
    * The Recognizaing Textual Entailment
    * MNLI task와 유사하지만 neutral과 contradiction을 모두 not-entailment로 바꿔서 이진 분류 문제로 변경
* __WNLI__
    * Winograd NLI의 줄임말
    * 모델이 대명사가 포함된 문장을 읽고, 대명사가 가리키는 대상을 리스트에서 고르는 task
    * 대명사가 포함된 문장이 원래 문장의 entailment인지 예측하는 것.