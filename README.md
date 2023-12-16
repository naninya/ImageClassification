# 紹介
 複数のModelingを利用し、ImageClassificationを行う

# docker 構造
classification - イメージ分類 

# 起動手順
1. git clone  
2. (./classification のdirectoryに移動)  
3. $docker-compose up  （GPU環境ではないとDockerが起動しません。） 
4. データを入れる  
(1)./data/{project_name}/train/images --> 学習イメージ  
(2)./data/{project_name}/train/annotations --> セグメンテーションマスク  
(3)./data/{project_name}/test/images --> テストイメージ   
(4)./data/{project_name}/test/annotations --> セグメンテーションマスク  
5. 学習、テスト  
*command lineで行う場合   
(1) docker exec -it classification .bash  
(2) python ./src/train.py  
*notebookで行う場合  
(1) {domain}:8028に接続
(2) ./classification/notebook/Sample_ImageClassification.ipynb を実行  
6. 実行結果  
./result/{project_name}/weights/{scheme}.ckpt --> 学習Model weight  
./result/{project_name}/result.pkl --> テストデータ結果  
