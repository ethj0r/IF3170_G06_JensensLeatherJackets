# IF3170_G06_JensensLeatherJackets

Implementasi Tugas Besar 1 IF3170 Inteligensi Artifisial untuk optimasi
penempatan paket tiga dimensi menggunakan Hill-Climbing, Simulated Annealing,
dan Genetic Algorithm.

## Menjalankan program

Siapkan virtual environment dan pasang dependency dari root repository:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

Jalankan seluruh automated test:

```bash
cd src
../.venv/bin/pytest -q
cd ..
```

## Menjalankan eksperimen melalui interface

Dari root repository, aktifkan virtual environment dan buka aplikasi:

```bash
source .venv/bin/activate
python -m streamlit run src/app/main.py
```

Buka alamat yang ditampilkan terminal, biasanya `http://localhost:8501`.
Interface menyediakan dua mode:

- **Eksperimen baru**: pilih dataset, algoritma, dan objective, isi seed serta
  parameter, lalu klik **Jalankan eksperimen**. Satu klik menjalankan satu
  percobaan. Mengubah input saja tidak menjalankan algoritma.
- **Hasil tersimpan**: buka kembali hasil JSON di `output/`, termasuk hasil
  dalam subfolder.

Pilihan algoritma mencakup Steepest Ascent Hill Climbing, Stochastic Hill
Climbing, Hill Climbing dengan sideways move, Random Restart Hill Climbing,
Simulated Annealing, Genetic Algorithm, dan random baseline. Input parameter
menyesuaikan algoritma yang dipilih. Objective yang tersedia adalah
`total_value` dan `priority_value`.

### Contoh input Simulated Annealing untuk laporan

Pilih mode **Eksperimen baru**, lalu isi:

| Input | Nilai |
| --- | --- |
| Dataset | `large` |
| Algoritma | Simulated Annealing |
| Objective | `total_value` |
| Seed | `0` |
| Temperatur awal | `100` |
| Faktor pendinginan (alpha) | `0.99982` |
| Temperatur minimum | `0.01` |
| Maksimum iterasi | `100` |
| Bobot penalti | `760` |
| Stuck threshold | `500` |

Klik **Jalankan eksperimen** dan ambil screenshot setelah hasil muncul.
Untuk percobaan berikutnya, ubah seed menjadi `1` atau `2`, lalu klik tombol
tersebut lagi. Ulangi langkah yang sama dengan maksimum iterasi `500` dan
`1000` untuk memperoleh sembilan percobaan, masing-masing dijalankan manual.

Hasil baru otomatis disimpan di `output/` dengan nama unik agar percobaan
sebelumnya tidak tertimpa. Definisi problem ikut disimpan bersama hasil.
Visualisasi dan metrik tetap tampil setelah input diubah sampai percobaan
berikutnya dijalankan; periksa judul hasil untuk memastikan seed dan dataset
yang sedang ditampilkan.

Interface menampilkan initial state, final state, seluruh paket di dalam maupun
di luar truk, grafik objective, parameter eksperimen, dan informasi khusus tiap
algoritma. SA dilengkapi grafik probabilitas penerimaan dan temperatur. Untuk
problem multi-truk tersedia satu tab untuk setiap truk.

## Menjalankan eksperimen melalui terminal

Sebagai alternatif, jalankan satu eksperimen Genetic Algorithm:

```bash
python src/run_experiments.py \
  --problem src/data/large.json \
  --algo ga \
  --population-size 100 \
  --iterations 500 \
  --seed 0
```

Pilihan algoritma untuk `--algo` adalah `hc`, `hc_stochastic`, `hc_sideways`,
`hc_restart`, `sa`, dan `ga`. Objective default adalah `total_value`; objective
bonus dapat dipilih dengan `--objective priority_value`.

Jalankan 24 eksperimen minimum sesuai spesifikasi:

```bash
python src/run_experiments.py \
  --problem src/data/large.json \
  --all
```

Rangkaian tersebut menjalankan tiga Steepest Ascent Hill-Climbing, tiga
Simulated Annealing, sembilan GA untuk variasi iterasi, dan sembilan GA untuk
variasi populasi. Hasil JSON disimpan pada folder `output/` dan perlu disertakan
dalam repository agar eksperimen dapat diperiksa serta divisualisasikan.

## Format input multi-truk

Armada heterogen dapat ditulis menggunakan daftar `trucks`:

```json
{
  "name": "multi_truck",
  "trucks": [
    { "id": "T01", "w": 10, "l": 20, "h": 5, "max_capacity": 500 },
    { "id": "T02", "w": 8, "l": 16, "h": 5, "max_capacity": 350 }
  ],
  "packages": []
}
```

Untuk beberapa truk identik, gunakan satu objek `truck` dan `truck_count`:

```json
{
  "name": "identical_fleet",
  "truck": { "w": 10, "l": 20, "h": 5, "max_capacity": 500 },
  "truck_count": 3,
  "packages": []
}
```

Format satu-truk lama tetap didukung tanpa perubahan.

Contoh eksperimen bonus multi-truk:

```bash
python src/run_experiments.py \
  --problem src/data/multi_truck.json \
  --algo ga \
  --population-size 100 \
  --iterations 500 \
  --seed 0
```

## Pembagian tugas

| NIM | Nama | Tugas utama |
| --- | --- | --- |
| 13524026 | Made Branenda Jordhy | Arsitektur awal, model dan neighbor, Simulated Annealing, serta visualisasi |
| 13524031 | Vincent Rionarlie | Hill-Climbing beserta seluruh variannya dan integrasi parameter eksperimen |
| 13524067 | Bryan Pratama Putra Hendra | Constraint dan packing, Genetic Algorithm, runner eksperimen, objective tambahan, serta multi-truk |

Seluruh anggota berkontribusi dalam pengujian, integrasi, penulisan laporan, dan
evaluasi hasil eksperimen.
