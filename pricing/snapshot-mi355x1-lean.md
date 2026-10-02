# MI355X 1x lean snapshot (OS + pulled ROCm image; scratch excluded)

- Snapshot ID: <snapshot-id>
- Name: <resource-name>
- Status: available
- size_gigabytes: 72.2
- min_disk_size_gb: 720
- regions: ['mem1']
- created_at: 2026-10-01T09:23:13Z
- Source Droplet ID: <droplet-id>
- Boot disk used at snapshot: ~68G of 697G root (/mnt wiped; /scratch = DOSCRATCH vdc1, not in snapshot)
- Docker image: rocm/pytorch:rocm7.14.1_ubuntu24.04_py3.12_pytorch_release_2.12.0 (~49.7GB via containerd) plus DOCR-tagged twin
- Snapshot storage rate (DigitalOcean published): $0.06 / GiB / month
- Estimated monthly cost: $4.33
- Replaces fat snapshot <snapshot-id> (334.46 GiB, ~$20.07/mo)
- Recorded: 2026-10-01T09:29:31.403743+00:00
