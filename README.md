# IF3170_G06_JensensLeatherJackets
Built for Tucil 1 IF3170 Artificial Intelligence

## Menjalankan program

Aktifkan virtual environment dan pasang dependency:

```bash
source .venv/bin/activate
pip install -r src/requirements.txt
```

Jalankan satu eksperimen Genetic Algorithm pada testcase multi-truk:

```bash
python src/run_experiments.py \
  --problem src/data/multi_truck.json \
  --algo ga \
  --population-size 100 \
  --iterations 500 \
  --seed 0
```

Jalankan visualizer:

```bash
streamlit run src/app/main.py
```

Hasil eksperimen disimpan pada folder `output/`. Untuk problem multi-truk,
visualizer menyediakan satu tab untuk setiap truk dan menampilkan initial state,
final state, penggunaan kapasitas, serta assignment setiap paket.

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
