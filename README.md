# Implementasi Sequential LSTM Model dalam Chatbot Kesehatan Mental Remaja Menggunakan TensorFlow

![image](https://github.com/Mazcho/Implementasi-Sequential-LSTM-Model-dalam-Chatbot-Kesehatan-Mental-Remaja-Menggunakan-TensorFlow/assets/77985996/5a78004d-a7f6-4da2-8a12-1dae7ad5927d)

## Latar Belakang Projek
Pada tahun 2023, kesehatan mental merupakan masalah terbesar bagi kaum remaja . Banyak kaum remaja yang mengalami berbagai masalah akan tetapi tidak ada yang bisa mendengarkan ceritanya, bahkan di keluarganya sendiri. Tidak semua orang mudah untuk menceritakan apa yang terjadi pada dirinya terhadap orang sekitarnya. Maka dari itu , menggunakan Machine Learning (ML) dan Deep Learning untuk membuat chatbot yang dapat merespon chat dari seseorang yang sedang mengalami masalah. PENDAHULUAN Latar Belakang Kesehatan mental adalah kondisi seorang individu terbebas dari segala bentuk gejala-gejala gangguan mental(1). Pada pertumbuhan di masa remaja, banyak terjadi perubahan biologis, psikis, maupun dari segi sosial(2). Kesehatan jiwa terdiri dari beberapa jenis kondisi yang secara umum dikategorikan dalam kondisi sehat, gangguan kecemasan, stres dan depresi (3). Tidak semua orang dapat mengungkapkan isi di otaknya di depan umum, maupun orang terdekatnya, karena mengalami trust issue. Maka dari itu, penulis ingin membuat sebuah chat bot bantuan kesehatan mental remaja, yang dimana chat bot ini setidaknya bisa membantu remaja merasa lebih tenang, karena pada dasarnya tidak smua orang bisa menceritakaan apa yang dia rasakan (4).


## Deskripsi Proyek
Proyek ini merupakan implementasi Sequential LSTM Model dalam pembuatan Chatbot untuk mendukung kesehatan mental remaja. Proyek ini menggunakan TensorFlow sebagai kerangka kerja utama untuk pengembangan model.

## Daftar Isi
- [Dataset](#dataset)
- [Tujuan](#tujuan)
- [Model dan Tahapan](#model-dan-tahapan)
- [Uji Hasil](#uji-hasil)
- [Deployment](#deployment)

## Dataset
  Deskripsi singkat tentang dataset yang digunakan dalam proyek ini. Jelaskan sumber data, format, dan detail karakteristiknya.


### Pengumpulan Data Set
![Alt text](image.png)
Figure 1.0 Datasets from Kaggle : ELVIS
To retrieve the data, I took it from a public data site, namely from Kaggle. This page presents data in .json format which contains general chat and knowledge regarding mental health. Theres the link forDataset : https://www.kaggle.com/datasets/elvis23/mental-health-conversational-data/data

### Dataset Features
![Alt text](image-1.png)

Figure 2.0 Datasets Features
In this data set there are 3 features. These include tags, patterns, and responses. Tags are markers of text data to mark the text as belonging to a category (greetings, greetings in the morning, afternoon, evening, asking for help, etc.). Pattern is a pattern given by the user for that category. When the user types good morning, he will enter the tag marker in the learning machine later. And the last one is responses, the response here is providing a response to input from the user regarding what the user provides from the computer. And the target of this research is tags, and the predictor variables are patterns and responses 

## Tujuan
  Tujuan utama dari proyek ini adalah menciptakan chatbot kesehatan mental remaja yang dapat memberikan dukungan dan informasi yang relevan untuk pengguna. Jelaskan secara singkat mengapa proyek ini penting dan apa yang diharapkan dicapai.

## Model dan Tahapan
  Deskripsikan arsitektur model Sequential LSTM yang digunakan dalam chatbot kesehatan mental remaja. Jelaskan tahapan pengembangan model, termasuk preprocessing data, pelatihan model, dan evaluasi.

## Uji Hasil
  Sajikan hasil uji dari model yang dikembangkan. Jelaskan metrik evaluasi yang digunakan dan berikan insight tentang kinerja model.

## Deployment
  Instruksi singkat atau tautan ke dokumentasi untuk melakukan deployment model chatbot. Jelaskan persyaratan sistem dan langkah-langkah yang diperlukan untuk mengintegrasikan chatbot ke dalam aplikasi atau platform yang diinginkan.

## Kontribusi
  Jika ada kontribusi yang diinginkan dari komunitas, jelaskan bagaimana orang dapat berkontribusi pada proyek ini.

## Lisensi
  Tentukan jenis lisensi yang digunakan untuk proyek ini. Jika memungkinkan, sertakan teks lengkap dari lisensi atau tautan ke file lisensi.

## Kontak
  Informasi kontak untuk pertanyaan atau kerjasama. Sertakan alamat email atau tautan media sosial yang relevan.

---

**Catatan:** Gantilah bagian-bagian yang diperlukan dengan informasi spesifik tentang proyek Anda.
