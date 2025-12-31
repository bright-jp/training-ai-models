# AIモデル トレーニングガイド

[![Bright Data Promo](https://github.com/bright-jp/LinkedIn-Scraper/raw/main/Proxies%20and%20scrapers%20GitHub%20bonus%20banner.png)](https://brightdata.jp/)

このガイドでは、OpenAIのツールセットによるファインチューニングを通じてAIモデルのパフォーマンスを向上させる方法を解説します。

- [AIとモデル トレーニングの概要](#introduction-to-ai-and-model-training)
- [ファインチューニングの準備](#getting-ready-for-fine-tuning)
- [ステップバイステップのファインチューニング](#step-by-step-fine-tuning)
- [成功のための戦略](#strategies-for-success)

## AIとモデル トレーニングの概要

AIシステムは、学習や推論など、人間のような認知タスクを実行します。これらのモデルはアルゴリズムを用いてデータから予測を行い、機械学習により、明示的なプログラミングではなく経験を通じて改善できます。

AIのトレーニングは、子どもが学ぶ方法に似ています。パターンを観察し、予測を立て、間違いから学びます。モデルには将来の予測のためにパターンを認識できるようデータが与えられ、予測を既知の結果と比較することで性能が測定されます。

ゼロからモデルを作成するには、事前知識なしでパターン認識を教える必要があり、多大なリソースを要します。また、データが限られている場合、最適とは言えない結果になりがちです。

ファインチューニングは、一般的なパターンをすでに理解している事前学習済みモデルから開始し、それらを特化したデータセットで学習させます。このアプローチは通常、より少ないリソースでより良い結果をもたらし、特化データが限られている場合に最適です。

## ファインチューニングの準備

特化データセットでの追加学習によって既存モデルを強化することは、ゼロから構築するより有利に見えますが、成功するファインチューニングは複数の重要要素に依存します。

### モデル選定戦略

ファインチューニングのベースモデルを選ぶ際は、次の要素を考慮してください。

**タスクの整合性:** 目的と期待する機能を明確に定義してください。類似タスクで優れた性能を発揮するモデルを選択します。元の目的とターゲット用途の不一致は効果を下げる可能性があります。たとえば、テキスト生成には [GPT-3](https://openai.com/index/gpt-3-apps/) が有効な場合があり、テキスト分類には [BERT](https://huggingface.co/docs/transformers/en/model_doc/bert) や [RoBERTa](https://huggingface.co/docs/transformers/en/model_doc/roberta) の方が適していることがあります。

**モデルサイズと複雑性:** 能力とリソース要件の適切なバランスを見つけてください。大きいモデルほど複雑なパターンを捉えられますが、より多くの計算資源を必要とします。

**評価指標:** タスクに関連する性能測定を選択してください。分類プロジェクトでは精度が重視されることが多い一方、言語生成では [BLEU](https://medium.com/@priyankads/evaluation-metrics-in-natural-language-processing-bleu-dc3cfa8faaa5) や [ROUGE](https://medium.com/nlplanet/two-minutes-nlp-learn-the-rouge-metric-by-examples-f179cc285499) スコアが有益な場合があります。

**コミュニティとリソース:** コミュニティのサポートが強固で、実装リソースが豊富なモデルを選ぶとよいです。明確なファインチューニング手順があり、信頼できる事前学習済みチェックポイントが提供されているものを優先してください。

### データ取得と処理

データセットの品質と多様性は、ファインチューニング後のモデル性能に大きく影響します。以下の重要点を考慮してください。

**必要なデータの種類:** 必要なデータタイプはタスクとモデルの事前学習内容によって異なります。NLPタスクでは通常、本、記事、ソーシャルメディア、文字起こしなどのソースからのテキストが必要です。収集方法にはWebスクレイピング、アンケート、またはプラットフォームAPIなどがあります。たとえば、[web scraping with AI](https://brightdata.jp/blog/web-data/ai-web-scraping) は、多様で最新のデータを大量に収集する際に有用であることが示されています。

**データクリーニングとアノテーション:** クリーニングでは、無関係なコンテンツの除去、不足情報の処理、フォーマットの標準化を行います。アノテーションは、モデル学習のためにデータへラベルを付与することです。[Bright Data](/) のようなツールは、これらの重要なプロセスを効率化できます。

**多様なデータセットの取り込み:** 多様で代表性のあるデータセットにより、モデルは複数の視点から学習でき、より汎化された信頼性の高い予測を生成します。たとえば、映画レビュー向けの感情分析モデルをファインチューニングする場合、現実の分布を反映するように、さまざまな映画タイプ、ジャンル、感情レベルからのフィードバックを含めてください。

### トレーニング環境の構成

選択したAIモデルとフレームワークに適したハードウェアとソフトウェアがあることを確認してください。大規模言語モデルは通常、GPUによって提供される大きな計算能力を必要とします。

TensorFlowやPyTorchなどのフレームワークは、AIモデル トレーニングの標準的な選択肢です。関連するライブラリや依存関係をインストールすることで、ワークフローへのスムーズな統合が可能になります。たとえば、OpenAIが開発したモデルをファインチューニングする際には、[OpenAI API](https://openai.com/index/openai-api/) が必要になる場合があります。

## ステップバイステップのファインチューニング

ファインチューニングの基礎を押さえたところで、実用的な自然言語処理アプリケーションを見ていきましょう。

[OpenAI API for fine-tuning](https://platform.openai.com/docs/guides/fine-tuning) を使用して、事前学習済みモデルをファインチューニングします。現在、ファインチューニングは gpt-3.5-turbo-0125（推奨）、gpt-3.5-turbo-1106、gpt-3.5-turbo-0613、babbage-002、davinci-002、および実験的な gpt-4-0613 などのモデルで動作します。GPT-4のファインチューニングは引き続き実験的であり、対象ユーザーは [fine-tuning interface](https://platform.openai.com/finetune) からアクセスを申請できます。

### データセットの準備

研究[によれば](https://arxiv.org/abs/2304.03439)、GPT-3.5には分析的推論に制限があります。2022年に公開されたLaw School Admission Testの分析的推論問題（AR-LSAT）を用いて、この領域で `gpt-3.5-turbo` を強化しましょう。このデータセットは[公開されています](https://github.com/csitfun/LogiEval/blob/main/Data/ar_lsat.jsonl)。

ファインチューニング後のモデル品質は、トレーニングデータに直接依存します。各データセット例は、OpenAIの [Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create) の会話フォーマット（role、content、任意のnameフィールドを含むメッセージリスト）に従い、[JSONL](https://jsonlines.org/) ファイルとして保存する必要があります。

`gpt-3.5-turbo` をファインチューニングするために必要なフォーマットは次のとおりです。

```json
{"messages": [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
```

この構造には、`system`、`user`、`assistant` の各ロール間の会話を形成するメッセージリストが含まれます。`system` ロールのcontentは、ファインチューニングされたシステムの振る舞いを定義します。

以下は、AR-LSATデータセットから整形した例です。

![AR-LSAT dataset example](https://github.com/bright-jp/training-ai-models/blob/main/images/AR-LSAT-dataset-example.png)

重要なデータセット上の考慮事項は次のとおりです。

- OpenAIはファインチューニングに少なくとも10件の例を要求しており、`gpt-3.5-turbo` では50〜100件の例を推奨しています。最適な数はユースケースにより異なります。ハイパーパラメータ調整のために、トレーニングファイルと並行して検証ファイルを作成できます。
- ファインチューニングおよびファインチューニング済みモデルの使用には、ベースモデルにより異なるトークンベースの課金が発生します。詳細は [OpenAI's pricing page](https://openai.com/api/pricing) を参照してください。
- トークン制限は選択するモデルに依存します。gpt-3.5-turbo-0125では最大コンテキスト長が16,385であるため、各トレーニング例は16,385トークンに制限され、これを超える例は切り詰められます。OpenAIの [counting tokens notebook](https://cookbook.openai.com/examples/How_to_count_tokens_with_tiktoken.ipynb) を使用してトークン数を算出してください。
- OpenAIは、潜在的なエラーの特定とトレーニング/検証ファイルのフォーマット検証のための [Python script](https://cookbook.openai.com/examples/chat_finetuning_data_prep) を提供しています。

### APIアクセスの設定

OpenAIモデルのファインチューニングには、十分なクレジット残高を持つ [OpenAI developer account](https://platform.openai.com/docs/overview) が必要です。

APIアクセスを設定するには、次の手順に従ってください。

1. [OpenAI website](https://platform.openai.com/overview) でアカウントを作成します。

2. 「Settings」配下の「Billing」セクションからアカウントにクレジットを追加し、ファインチューニングを有効化します。

![Billing settings on OpenAI](https://github.com/bright-jp/training-ai-models/blob/main/images/Billing-settings-on-OpenAI-1-1024x562.png)

3. 左上のプロフィールアイコンをクリックし、「API Keys」を選択してキー作成ページに移動します。

![Accessing API keys in OpenAI's settings](https://github.com/bright-jp/training-ai-models/blob/main/images/Accessing-API-keys-in-OpenAIs-settings-1-1024x396.png)

4. 説明的な名前を入力して、新しいシークレットキーを生成します。

![Generating a new API key on OpenAI](https://github.com/bright-jp/training-ai-models/blob/main/images/Generating-a-new-API-key-on-OpenAI.png)

5. ファインチューニング機能を利用するために、OpenAIのPythonライブラリをインストールします。

```sh
pip install openai
```

6. 通信を確立するために、APIキーを環境変数として設定します。

```python
import os
from openai import OpenAI

# Set the OPENAI_API_KEY environment variable
os.environ['OPENAI_API_KEY'] = 'The key generated in step 4'
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
```

### トレーニング素材のアップロード

データを検証したら、ファインチューニングジョブ向けに [Files API](https://platform.openai.com/docs/api-reference/files/create) を使用してファイルをアップロードします。

```python
training_file_id = client.files.create(
  file=open(training_file_name, "rb"),
  purpose="fine-tune"
)
validation_file_id = client.files.create(
  file=open(validation_file_name, "rb"),
  purpose="fine-tune"
)
print(f"Training File ID: {training_file_id}")
print(f"Validation File ID: {validation_file_id}")
```

正常に実行されると、トレーニングおよび検証データセットの双方に一意の識別子が割り当てられます。

![unique identifiers example](https://github.com/bright-jp/training-ai-models/blob/main/images/unique-identifiers-example-1024x87.png)

### ファインチューニング セッションの開始

ファイルをアップロードしたら、[user interface](https://platform.openai.com/finetune) またはプログラムからファインチューニングジョブを作成します。

OpenAI SDKでファインチューニングジョブを開始する方法は次のとおりです。

```python
response = client.fine_tuning.jobs.create(
  training_file=training_file_id.id,
  validation_file=validation_file_id.id,
  model="gpt-3.5-turbo",
  hyperparameters={
    "n_epochs": 10,
    "batch_size": 3,
    "learning_rate_multiplier": 0.3
  }
)
job_id = response.id
status = response.status

print(f'Fine-tunning model with jobID: {job_id}.')
print(f"Training Response: {response}")
print(f"Training Status: {status}")
```

- `model`: ファインチューニングするモデルを指定します（`gpt-3.5-turbo`、`babbage-002`、`davinci-002`、または既存のファインチューニング済みモデル）。
- `training_file` と `validation_file`: アップロード時に返されたファイル識別子です。
- `n_epochs`、`batch_size`、`learning_rate_multiplier`: カスタマイズ可能なトレーニングパラメータです。

追加のファインチューニングオプションについては、[API documentation](https://platform.openai.com/docs/api-reference/fine-tuning/create) を参照してください。

このコードは、ジョブID `ftjob-0EVPunnseZ6Xnd0oGcnWBZA7` の情報を生成します。

![Example of the information generated by the code above](https://github.com/bright-jp/training-ai-models/blob/main/images/Example-of-the-information-generated-by-the-code-above.png)

ファインチューニングジョブは他のジョブの後ろにキューイングされるため、完了までにかなりの時間がかかる場合があります。トレーニング時間は、モデルの複雑性とデータセットサイズに応じて数分から数時間まで変動します。

完了すると、OpenAIからメールで確認が届きます。

ファインチューニングインターフェースでジョブステータスを監視してください。

### モデル性能の評価

OpenAIはトレーニング中に、以下の主要メトリクスを算出します。

- Training loss
- Training token accuracy
- Validation loss
- Validation token accuracy

検証メトリクスは2つの方法で計算されます。各ステップでの小さなデータバッチに対して計算する方法と、各エポック後に完全な検証セットに対して計算する方法です。完全な検証メトリクスが最も正確な性能指標を提供し、トレーニングが滑らかに進行していることを検証します（lossは減少し、token accuracyは増加するはずです）。

ファインチューニング実行中は、次の方法でこれらのメトリクスを確認できます。

**1. ユーザーインターフェース:**

![The fine tuning UI](https://github.com/bright-jp/training-ai-models/blob/main/images/The-fine-tuning-UI.png)

**2. API:**

```python
import os
from openai import OpenAI

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'],)
jobid = 'jobid you want to monitor'

print(f"Streaming events for the fine-tuning job: {jobid}")

# signal.signal(signal.SIGINT, signal_handler)

events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=jobid)
try:
    for event in events:
        print(
            f'{event.data}'
        )
except Exception:
    print("Stream interrupted (client disconnected).")
```

このコードは、ステップ番号、トレーニング/検証loss値、総ステップ数、トレーニング/検証セット双方の平均token accuracyを含むストリーミングイベントを出力します。

```
Streaming events for the fine-tuning job: ftjob-0EVPunnseZ6Xnd0oGcnWBZA7
{'step': 67, 'train_loss': 0.30375099182128906, 'valid_loss': 0.49169286092122394, 'total_steps': 67, 'train_mean_token_accuracy': 0.8333333134651184, 'valid_mean_token_accuracy': 0.8888888888888888}
```

### 結果の最適化

ファインチューニング結果が期待に届かない場合は、次の改善戦略を検討してください。

**1. データセットの改善:**

- 特定のモデルの弱点に対処する例を追加し、レスポンスの分布が期待されるパターンに合致していることを確認してください。
- 再現可能なデータの問題がないか確認し、例に適切なレスポンスに必要な情報がすべて含まれていることを検証してください。
- 複数の寄稿者によるデータの一貫性を維持し、推論時の期待に合わせて全例のフォーマットを標準化してください。
- 一般に、高品質なデータは、低品質な情報を大量に用意するよりも優れた成果をもたらすことを忘れないでください。

**2. パラメータの調整:**

- OpenAIでは、epochs、learning rate multiplier、batch sizeの3つの主要ハイパーパラメータをカスタマイズできます。
- まずはデータセット特性に基づいて組み込み関数が選択するデフォルト値から開始し、必要に応じて調整してください。
- モデルがトレーニングパターンに十分従わない場合は、epoch数を増やしてください。
- モデルの応答の多様性が不足する場合は、epochsを1〜2減らしてください。
- 収束の問題が生じた場合は、learning rate multiplierを増やしてください。

### チェックポイント済みモデルの利用

OpenAIは現在、各ファインチューニングジョブの最後の3エポック分のチェックポイントへのアクセスを提供しています。これらのチェックポイントは推論および追加のファインチューニングに利用できる完全なモデルです。

チェックポイントへアクセスするには、ジョブ完了を待ってから、ファインチューニングジョブIDを使用して [query the checkpoints endpoint](https://platform.openai.com/docs/api-reference/fine-tuning/list-checkpoints) を実行してください。各チェックポイントには、モデルチェックポイント名を含む `fine_tuned_model_checkpoint` フィールドがあります。ユーザーインターフェースからチェックポイント名を取得することもできます。

[**openai.chat.completions.create()**](https://platform.openai.com/docs/api-reference/chat) 関数を使用し、プロンプトとモデル名を指定してクエリを送信することで、チェックポイントモデルの性能をテストできます。

```python
completion = client.chat.completions.create(
  model="ft:gpt-3.5-turbo-0125:personal::9PWZuZo5",
  messages=[
    {"role": "system", "content": "Instructions: You will be presented with a passage and a question about that passage. There are four options to be chosen from, you need to choose the only correct option to answer that question. If the first option is right, you generate the answer 'A', if the second option is right, you generate the answer 'B', if the third option is right, you generate the answer 'C', if the fourth option is right, you generate the answer 'D', if the fifth option is right, you generate the answer 'E'. Read the question and options thoroughly and select the correct answer from the four answer labels. Read the passage thoroughly to ensure you know what the passage entails"},
    {"role": "user", "content": "Passage: For the school paper, five students\u2014Jiang, Kramer, Lopez, Megregian, and O'Neill\u2014each review one or more of exactly three plays: Sunset, Tamerlane, and Undulation, but do not review any other plays. The following conditions must apply: Kramer and Lopez each review fewer of the plays than Megregian. Neither Lopez nor Megregian reviews any play Jiang reviews. Kramer and O'Neill both review Tamerlane. Exactly two of the students review exactly the same play or plays as each other.Question: Which one of the following could be an accurate and complete list of the students who review only Sunset?\nA. Lopez\nB. O'Neill\nC. Jiang, Lopez\nD. Kramer, O'Neill\nE. Lopez, Megregian\nAnswer:"}
  ]
)
print(completion.choices[0].message)
```

answer dictionaryからのレスポンス:

![the result from the answer dictionary](https://github.com/bright-jp/training-ai-models/blob/main/images/the-result-from-the-answer-dictionary.png)

また、[OpenAI's playground](https://platform.openai.com/playground/) で、ファインチューニング済みモデルを他のモデルと比較することもできます。

![Example of the fine tuned model vs other models in the playground](https://github.com/bright-jp/training-ai-models/blob/main/images/Example-of-the-fine-tuned-model-vs-other-models-in-the-playground-1-1024x776.png)

## 成功のための戦略

効果的なファインチューニングのために、次の推奨事項を検討してください。

**データ品質:** 過学習（トレーニングデータでは良い性能だが新しい入力では悪い性能）を防ぐために、特化データがクリーンで多様かつ代表性があることを確認してください。

**パラメータ選定:** 収束が遅くなったり最適とは言えない結果になったりしないよう、適切なハイパーパラメータを選択してください。複雑で時間がかかる一方、このステップは効果的なトレーニングに不可欠です。

**リソース計画:** 大規模モデルのファインチューニングには、相当な計算能力と時間配分が必要であることを認識してください。

### 一般的な課題

**複雑性のバランス:** 過学習（過度な分散）と学習不足（過度なバイアス）を防ぐため、モデルの複雑性とトレーニング時間の適切な均衡点を見つけてください。

**知識の保持:** ファインチューニング中、モデルが既に獲得した一般知識を失う場合があります。この問題を軽減するため、さまざまなタスクにわたって定期的に性能をテストしてください。

**ドメイン適応:** ファインチューニングデータが事前学習データと大きく異なる場合、ドメインシフトの問題が生じる可能性があります。これらのギャップに対処するためにドメイン適応手法を実装してください。

### モデルの保存

トレーニング後は、将来の利用に備えてモデルの完全な状態（モデルパラメータとオプティマイザの状態を含む）を保存してください。これにより、後で同じ地点からトレーニングを再開できます。

### 倫理的含意

**バイアスへの懸念:** 事前学習済みモデルには固有のバイアスが含まれる可能性があり、ファインチューニングによってそれが増幅されることがあります。偏りのない予測が重要な場合は、公平性の観点でテストされた事前学習済みモデルを選択してください。

**出力の検証:** ファインチューニング済みモデルは、もっともらしいが誤った情報を生成する場合があります。こうしたケースに対処するため、堅牢な検証システムを実装してください。

**性能劣化:** 環境やデータ分布の変化により、モデルの性能が時間とともに低下することがあります。定期的に性能を監視し、必要に応じてファインチューニングを更新してください。

### 最先端の手法

LLM向けの高度なファインチューニング手法には、Low Ranking Adaptation（LoRA）やQuantized LoRA（QLoRA）があり、性能を維持しながら計算および金銭的要件を削減します。Parameter Efficient Fine Tuning（PEFT）は、最小限のパラメータ調整で効率的にモデルを適応させます。DeepSpeedとZeROは大規模トレーニングにおけるメモリ使用量を最適化します。これらの手法は、過学習、知識損失、ドメイン適応などの課題に対処し、LLMのファインチューニング効率と有効性を高めます。

ファインチューニングに加えて、転移学習や強化学習などの高度なトレーニング手法も検討してください。転移学習は、[あるドメインの知識を関連する問題に適用](https://brightdata.jp/blog/web-data/data-pitfalls-when-developing-ai-models) し、強化学習は、環境との相互作用と報酬最大化を通じてエージェントが最適な意思決定を学習できるようにします。

AIモデル トレーニングをさらに深く学ぶために、次の有用なリソースをご参照ください。

- [Attention is all you need by Ashish Vaswani et al.](https://arxiv.org/abs/1706.03762)
- [The book "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville](https://www.deeplearningbook.org/)
- [The book "Speech and Language Processing" by Daniel Jurafsky and James H. Martin](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)
- [Different ways of training LLMs](https://towardsdatascience.com/different-ways-of-training-llms-c57885f388ed)
- [Mastering LLM Techniques: Training](https://developer.nvidia.com/blog/mastering-llm-techniques-training/)
- [NLP course by Hugging Face](https://huggingface.co/learn/nlp-course/chapter1/1)

## 最後に

効果的なAIモデルの開発には、大量の高品質データが必要です。問題定義、モデル選定、反復的な改善は重要ですが、真の差別化要因はデータの品質と量にあります。

Webスクレイパーを自作して維持するのではなく、Bright Dataのプラットフォームで用意された事前構築済み、または[custom datasets](https://brightdata.jp/products/datasets) によりデータ収集を簡素化してください。

今すぐ登録して無料トライアルを開始しましょう！