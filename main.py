import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc


def veri_olustur():
    tablo = sns.load_dataset("titanic")
    tablo = tablo[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare"]]
    tablo.columns = ["Hayatta", "Sinif", "Cinsiyet", "Yas", "KardesSayisi", "EbeveynSayisi", "Ucret"]
    tablo["Cinsiyet"] = tablo["Cinsiyet"].map({"male": 1, "female": 0})
    tablo["Yas"] = tablo["Yas"].fillna(tablo["Yas"].median())
    print("Gerçek Titanic verisi yüklendi:", tablo.shape)
    return tablo


def veriyi_incele(tablo):
    print("\n─── VERİ ÖNİZLEME ──────────────────────────────────")
    print(tablo.head(10).to_string())

    print("\n─── BOYUT ──────────────────────────────────────────")
    print(f"  {tablo.shape[0]} satır, {tablo.shape[1]} sütun")

    print("\n─── EKSİK DEĞERLER ─────────────────────────────────")
    print(tablo.isnull().sum().to_string())

    print("\n─── İSTATİSTİKLER ──────────────────────────────────")
    print(tablo.describe().round(2).to_string())

    print("\n─── HAYATTA KALMA DAĞILIMI ─────────────────────────")
    dagilim = tablo["Hayatta"].value_counts()
    print(f"  Hayatta kalmadı (0): {dagilim[0]} kişi")
    print(f"  Hayatta kaldı   (1): {dagilim[1]} kişi")
    print(f"  Hayatta kalma oranı: %{dagilim[1] / len(tablo) * 100:.1f}")
    print("────────────────────────────────────────────────────")


def veriyi_bol(tablo):
    X = tablo[["Sinif", "Cinsiyet", "Yas", "KardesSayisi", "EbeveynSayisi", "Ucret"]]
    y = tablo["Hayatta"]
    X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Eğitim: {len(X_egitim)} | Test: {len(X_test)}")
    return X_egitim, X_test, y_egitim, y_test


def modelleri_egit(X_egitim, y_egitim):
    ka = DecisionTreeClassifier(max_depth=5, random_state=42)
    ro = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    ka.fit(X_egitim, y_egitim)
    ro.fit(X_egitim, y_egitim)
    print("Modeller eğitildi")
    return ka, ro


def sonuclari_goster(ka, ro, X_test, y_test):
    tahmin_ka = ka.predict(X_test)
    tahmin_ro = ro.predict(X_test)

    for model_adi, tahmin in [("KARAR AĞACI", tahmin_ka), ("RASTGELE ORMAN", tahmin_ro)]:
        print(f"\n─── {model_adi} ────────────────────────────────────")
        oran = accuracy_score(y_test, tahmin)
        print(f"  Doğruluk: %{oran * 100:.1f}")
        print("\n  Detaylı Metrikler:")
        print(classification_report(y_test, tahmin,
                                    target_names=["Hayatta Kalmadı", "Hayatta Kaldı"],
                                    digits=3))
        cm = confusion_matrix(y_test, tahmin)
        print("  Karışıklık Matrisi:")
        print(f"                 Tahmin:Kalmadı  Tahmin:Kaldı")
        print(f"  Gerçek:Kalmadı      {cm[0][0]:>5}          {cm[0][1]:>5}")
        print(f"  Gerçek:Kaldı        {cm[1][0]:>5}          {cm[1][1]:>5}")
        print("──────────────────────────────────────────────────")

    oran_ka = accuracy_score(y_test, tahmin_ka)
    oran_ro = accuracy_score(y_test, tahmin_ro)
    kazanan = "Karar Ağacı" if oran_ka > oran_ro else "Rastgele Orman"
    print(f"\nKazanan: {kazanan}")


def grafikleri_goster(ka, ro, X_test, y_test):
    tahmin_ka   = ka.predict(X_test)
    tahmin_ro   = ro.predict(X_test)
    olasilik_ka = ka.predict_proba(X_test)[:, 1]
    olasilik_ro = ro.predict_proba(X_test)[:, 1]

    fig, eksenler = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Karar Ağacı vs Rastgele Orman", fontsize=16, fontweight="bold")

    ax = eksenler[0, 0]
    dogruluklar = [accuracy_score(y_test, tahmin_ka) * 100,
                   accuracy_score(y_test, tahmin_ro) * 100]
    ax.bar(["Karar Ağacı", "Rastgele Orman"], dogruluklar,
           color=["#E07B54", "#4C8BE0"], width=0.5)
    ax.set_title("Doğruluk Karşılaştırması")
    ax.set_ylabel("Doğruluk (%)")
    ax.set_ylim(60, 100)
    for i, val in enumerate(dogruluklar):
        ax.text(i, val + 0.3, f"%{val:.1f}", ha="center", fontweight="bold")

    ax = eksenler[0, 1]
    sns.heatmap(confusion_matrix(y_test, tahmin_ka), annot=True, fmt="d",
                cmap="Oranges", ax=ax,
                xticklabels=["Kalmadı", "Kaldı"],
                yticklabels=["Kalmadı", "Kaldı"])
    ax.set_title("Karışıklık Matrisi — Karar Ağacı")
    ax.set_ylabel("Gerçek")
    ax.set_xlabel("Tahmin")

    ax = eksenler[1, 0]
    sns.heatmap(confusion_matrix(y_test, tahmin_ro), annot=True, fmt="d",
                cmap="Blues", ax=ax,
                xticklabels=["Kalmadı", "Kaldı"],
                yticklabels=["Kalmadı", "Kaldı"])
    ax.set_title("Karışıklık Matrisi — Rastgele Orman")
    ax.set_ylabel("Gerçek")
    ax.set_xlabel("Tahmin")

    ax = eksenler[1, 1]
    for olasilik, renk, isim in [(olasilik_ka, "#E07B54", "Karar Ağacı"),
                                  (olasilik_ro, "#4C8BE0", "Rastgele Orman")]:
        fpr, tpr, _ = roc_curve(y_test, olasilik)
        ax.plot(fpr, tpr, color=renk, lw=2, label=f"{isim} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Rastgele tahmin")
    ax.set_title("ROC Eğrisi")
    ax.set_xlabel("Yanlış Pozitif Oranı")
    ax.set_ylabel("Doğru Pozitif Oranı")
    ax.legend()

    plt.tight_layout()
    plt.savefig("grafik.png", dpi=150, bbox_inches="tight")
    print("Grafik kaydedildi: grafik.png")
    plt.close()


def agaci_goster(ka, X_egitim):
    fig, ax = plt.subplots(figsize=(30, 12))
    plot_tree(ka,
              feature_names=list(X_egitim.columns),
              class_names=["Kalmadı", "Kaldı"],
              filled=True,
              rounded=True,
              fontsize=9,
              ax=ax,
              impurity=True,
              proportion=False,
              precision=3)
    fig.suptitle("Karar Ağacı Yapısı", fontsize=18, fontweight="bold")
    plt.tight_layout()
    plt.savefig("karar_agaci.png", dpi=150, bbox_inches="tight")
    print("Karar ağacı kaydedildi: karar_agaci.png")
    plt.close()

def main():
    tablo = veri_olustur()
    veriyi_incele(tablo)
    X_egitim, X_test, y_egitim, y_test = veriyi_bol(tablo)
    ka, ro = modelleri_egit(X_egitim, y_egitim)
    sonuclari_goster(ka, ro, X_test, y_test)
    grafikleri_goster(ka, ro, X_test, y_test)
    agaci_goster(ka, X_egitim)

if __name__ == "__main__":
    main()