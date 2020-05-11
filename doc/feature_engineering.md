# For Long Tail Features
* **log => click_times**: <br>
    top1 - 1(0.945), <br>
    top2 - 2(0.05), <br>
    top3 - 3/4(0.004), <br>
    top4 - 5/6/7/8(0.0004), <br>
    top5 - 9-15, <br>
    top6 - 16-20, <br>
    top7 - 21-30, <br>
    top8 - 31-40, <br>
    top9 - 41-60, <br>
    top10 - >60<br>
* **user => age**: <br>
    top1 - 2/3/4(0.56), <br>
    top2 - 1(0.04), <br>
    top3 - 5/6/7(0.33), <br>
    top4 - 8/9/10(0.12)<br>
* **ads => product_category**: <br>
    top1 - 2(0.37), <br>
    top2 - 18(0.27), <br>
    top3 - 3/5/8(0.30), <br>
    top4 - 13/17/4/12/6(0.05), <br>
    top5 - 7/16/9/11(0.008), <br>
    top6 - 1/15, <br>
    top7 - 10/14<br>
* **ads => industry**: <br>
    top1 - 247/319/6/322/0(>100000, 0.35), <br>
    top2 - 242, 238, 326,  54,  73, 248, 317, 329,  25,  36,  47,  27,  21, 259, 133,  60, 297,  40, 253, 246,  13,  24, 252,  26, 289,  74, 296,  28, 207, 215,  34, 328, 216, 302, 200, 202,  88, 300, 217, 277, 176,  84,   5, 203,  86, 147(10000<x<100000, [5:51], 0.52)<br>
    top3 - (1000<x<10000, [52:131], 0.11)<br>
    top4 - (100<x<1000, [131:201], 0.013)<br>
    top5 - (x<100, [201:], 0.001)<br>
# For Stat Features (main key: user_id)
### importance: product_category > industry > product_id >  advertiser_id > ad_id > creative_id
### count
* ad_id_count
* ad_id_

* product_category_all
* product_category_count_top1
* product_category_count_top3
### overall count - key-value memory
* **overall count + product_category**
* overall count + industry
* overall count + product_id
* overall count + advertiser_id
* overall count + ad_id
* overall count + creative_id
### time range!!!
* **range + product_category**
* range + industry
* range + product_id
* range + advertiser_id
* range + ad_id
* range + creative_id
### click rate (user_single_ad_click/user_click_count) - key-value memory
* **rate + product_category**
* rate + industry
* rate + product_id
* rate + advertiser_id
* rate + ad_id
* rate + creative_id
### click count - key-value memory
* **count + product_category**
* count + industry
* count + product_id
* count + advertiser_id
* count + ad_id
* count + creative_id
### second order key-value memory
# For Embedding
### word2vec
* **product_category_embedding**
* **industry_embedding**
* **advertiser_id_embedding**
* **product_id_embedding**
* ad_id_embedding
* creative_id_embedding
<br><br>
* ad_id_creative_id_embedding
* **industry_advertiser_id_embedding**
* **product_category_product_id_embedding**

### NOTE: add embedding layer for primary-foreign pair specifically
### general word2vec (to be extended - time range) 
* **product_category_embedding_count_top3**
* **product_category_embedding_count_top6**
<br><br>
* **industry_embedding_count_top3**
* **industry_embedding_count_top6**
* **industry_embedding_count_top10**
<br><br>
* **advertiser_id_embedding_count_top2**
* **advertiser_id_embedding_count_top5**
* **advertiser_id_embedding_count_top10**
<br><br>
(contain null)
* **product_id_embedding_count_top5**
* **product_id_embedding_count_top9**
* **product_id_embedding_count_top15**
<br><br>
* ad_id_embedding_count_top2
* ad_id_embedding_count_top8
<br><br>
* creative_id_embedding_count_top6
* creative_id_embedding_count_top10
### field-aware embedding
* ad_id_creative_id_field_embedding
* **industry_advertiser_id_field_embedding**
* **product_category_product_id_field_embedding**
### graph embedding (DeepWalk)
* product_category_user_deepwalk_embedding
* **industry_user_deepwalk_embedding**
* **advertiser_id_user_deepwalk_embedding**
* **product_id_user_deepwalk_embedding**
* ad_id_user_deepwalk_embedding
* creative_id_user_deepwalk_embedding
### graph encoding (GCN)
* product_category_user_gcn_embedding
* **industry_user_gcn_embedding**
* **advertiser_id_user_gcn_embedding**
* **product_id_user_gcn_embedding**
* ad_id_user_gcn_embedding
* creative_id_user_gcn_embedding
# Embedding x Time
### oriented graph embedding (DeepWalk)
* product_category_user_time_deepwalk_embedding
* **industry_user_time_deepwalk_embedding**
* **advertiser_id_user_time_deepwalk_embedding**
* **product_id_user_time_deepwalk_embedding**
* ad_id_user_time_deepwalk_embedding
* creative_id_user_time_deepwalk_embedding