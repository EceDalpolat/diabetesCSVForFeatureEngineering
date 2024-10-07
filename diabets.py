import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
def load():
    data = pd.read_csv("diabetes/diabetes.csv")
    return data


df = load()
df.head()
##histogramda kullanılabilir
sns.boxplot(x=df["Age"], y=df["Outcome"])
plt.show()
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    result = check_outlier(df, col)
    print(f'{col} sütunu için sonuç: {result}')

for col in cat_cols:
    print(f"{col} değişkenine göre {"Outcome"} ortalaması:")
    print(df.groupby("Outcome")[col].mean(), "\n")

for col in num_cols:
    print(f"{col} değişkenine göre {"Outcome"} ortalaması:")
    print(df.groupby("Outcome")[col].mean(), "\n")


# Eksik gözlem sayısını bulalım
missing_values = df.isnull().sum()

# Eksik gözlem oranını hesaplayalım
missing_ratio = (df.isnull().sum() / len(df)) * 100

# Eksik değerleri hem sayısal hem de oransal olarak bir tablo halinde görelim
missing_data = pd.DataFrame({'Missing Values': missing_values, 'Missing Ratio (%)': missing_ratio})

# Eksik değeri olan sütunları görelim
missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values(by='Missing Values', ascending=False)

print(missing_data)


# Sayısal değişkenlerin korelasyon matrisini oluşturalım
correlation_matrix = df.corr()

# Korelasyon matrisini görselleştirmek için
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Korelasyon Matrisi')
plt.show()

# Yüksek korelasyonlu değişkenleri bulalım
threshold = 0.9  # 0.9 üzerindeki korelasyonlara bakalım
high_correlation = correlation_matrix[(correlation_matrix >= threshold) | (correlation_matrix <= -threshold)]
print(high_correlation)

print(df.dtypes)  # Her sütunun veri tipini kontrol eder
# Eksik değerlerin sayısını kontrol edelim
print(df.isnull().sum())


# Sabit sütunları kontrol edelim
constant_columns = [col for col in df.columns if df[col].nunique() == 1]
print("Sabit sütunlar:", constant_columns)

# Veri setinizin genel istatistiklerine bakalım
print(df.describe())

"""SkinThickness (Deri Kalınlığı), Insulin, ve BloodPressure (Kan Basıncı) sütunlarının minimum değerleri 0 
olarak gözüküyor. Ancak bu sütunların sıfır değeri alması tıbbi açıdan pek olası değil. Örneğin, bir kişinin 
deri kalınlığının veya kan basıncının sıfır olması imkansız. Bu sütunlarda aykırı değerler ya da eksik verilerin 
sıfırlarla doldurulmuş olma olasılığı yüksek.
Insulin değerlerinin 0 olması da benzer şekilde anormal. Bu sütunlarda eksik değerleri uygun bir yöntemle 
doldurmak gerekebilir.

Glucose sütunu (kan şekeri seviyesi) de sıfır değer içeriyor ki bu da biyolojik olarak pek mümkün değil.
Glucose sıfır olan verilerde eksik gözlemler olabilir.

BMI sütununda da sıfır değerler var. Bu da yine tıbbi olarak anlamlı olmayabilir, çünkü hiçbir insanın BMI'si 
0 olamaz. Bu değerler muhtemelen eksik veri olarak değerlendirilmelidir.
"""
# Sıfır değeri mantıksız olan sütunları seçiyoruz
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Bu sütunlarda sıfır olan değerleri NaN ile değiştirelim
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Eksik gözlemleri kontrol edelim
print(df.isnull().sum())

# Eksik değerleri (NaN) ortalama ile dolduralım
df[cols_with_zero] = df[cols_with_zero].fillna(df[cols_with_zero].mean())

# Eksik değerler tekrar kontrol edilebilir
print(df.isnull().sum())

# Aykırı değerleri düzenleyelim
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

"""for col in num_cols:
    new_df = remove_outlier(df, col)"""

df_cleaned = remove_outlier(df, "Insulin")

# Temizlenmiş veri setini görüntüleyelim
print(df_cleaned.shape)


# Korelasyon matrisini hesaplayalım
corr_matrix = df_cleaned.corr()

# Korelasyon matrisini görselleştirelim
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Korelasyon Matrisi')
plt.show()

threshold = 0.9  # 0.9 üzerindeki korelasyonlara bakalım
high_correlation = corr_matrix[(corr_matrix >= threshold) | (corr_matrix <= -threshold)]
print(high_correlation)

# BMI kategorisi oluşturma
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif 18.5 <= bmi < 24.9:
        return 'Normal weight'
    elif 25 <= bmi < 29.9:
        return 'Overweight'
    else:
        return 'Obesity'

df['BMI_Category'] = df['BMI'].apply(categorize_bmi)

# Oluşturulan yeni değişkenleri görüntüleyeli
df.head()

##Asıl veri setinde kategorik değişkenimiz bulunmamakta ama değişken oluşturma yöntemi ile oluşturduğumuz BMI_Category
#değişkenimiz kategorik ona one-hot uygulayalım drop_ffirt unutma
# Kategorik değişkenleri one-hot encoding ile dönüştürme
df_encoded = pd.get_dummies(df, columns=['BMI_Category'], drop_first=True)

# Yeni veri çerçevesini görüntüleyelim
df_encoded.head()

from sklearn.preprocessing import StandardScaler

# Standartlaştırma için scaler oluşturma
scaler = StandardScaler()

# Standartlaştırmayı uygulama
df_encoded[num_cols] = scaler.fit_transform(df_encoded[num_cols])

# Standardize edilmiş veri çerçevesini görüntüleyelim
print(df_encoded[num_cols].head())

##model kuralım targetımız sınıflandırma problemi olduğu için sınıflandırma algoritması olan logisticregresnyon uyguladım.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Özellikler ve hedef değişkeni tanımlama
X = df_encoded.drop(['Outcome'], axis=1)
y = df_encoded['Outcome']

# Eğitim ve test setine ayırma %20 test %80 eğitim olarak ayırdım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluşturma
model = LogisticRegression()
model.fit(X_train, y_train)

# Tahmin yapma
y_pred = model.predict(X_test)

# Modelin performansını değerlendirme
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
