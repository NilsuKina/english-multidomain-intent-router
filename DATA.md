# Veri kaynağı ve kullanım notu

- **Adı:** CLINC150 (`data_full.json`), İngilizce
- **İçerik:** 150 bilinen niyet ve ayrı `out-of-scope` örnekleri; resmî train/validation/test bölümleri korunur.
- **Full split boyutları:** 15,100 train, 3,100 validation, 5,500 test (toplam 23,700); testte 1,000 OOS örneği bulunur. Kaynak: [UCI CLINC150 kaydı](https://archive.ics.uci.edu/dataset/570/clinc150).
- **Kaynak:** [CLINC oos-eval](https://github.com/clinc/oos-eval), [UCI dataset record](https://archive.ics.uci.edu/dataset/570/clinc150)
- **Lisans:** UCI kaydında CC BY 4.0 olarak belirtilmiştir. Sunum ve README'de kaynak atfı yapılmalı.
- **Makale:** Larson et al. (2019), “An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction,” EMNLP-IJCNLP.

CLINC150, hocanın önerdiği İngilizce intent-classification veri setidir ve out-of-scope değerlendirmesini destekler. Örnekler genel sanal asistan istekleridir; bu yüzden proje CLINC150'nin çok alanlı asistan niyetlerini ilgili demo ekiplerine yönlendiren bir sistem olarak çerçevelenmiştir.
