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

Jalankan satu eksperimen Genetic Algorithm:

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

Jalankan visualizer:

```bash
streamlit run src/app/main.py
```

Visualizer menampilkan initial state, final state, seluruh paket di dalam maupun
di luar truk, grafik objective, parameter eksperimen, dan informasi khusus tiap
algoritma. Untuk problem multi-truk tersedia satu tab untuk setiap truk.

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
