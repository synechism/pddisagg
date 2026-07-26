SHELL := /bin/bash

.PHONY: phase0-create phase0-status phase0-ssh phase0-copy phase0-bootstrap \
	phase0-verify phase0-bandwidth phase0-serve phase0-smoke phase1-copy \
	phase1-smoke-load phase1-baseline phase1-pull phase1-aggregate \
	loadgen-create loadgen-status loadgen-copy loadgen-bootstrap \
	phase1-baseline-network phase1-pull-network phase1-aggregate-network \
	decode-create decode-bootstrap pd-workers-start pd-router-start pd-ici-start \
	pd-status

phase0-create:
	./scripts/cloud/create_phase0.sh

phase0-status:
	./scripts/cloud/status_phase0.sh

phase0-ssh:
	gcloud compute tpus tpu-vm ssh "$${TPU_NAME:-pd-phase0-v5e-4}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}"

phase0-copy:
	gcloud compute tpus tpu-vm ssh "$${TPU_NAME:-pd-phase0-v5e-4}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='mkdir -p ~/pd-disagg-remote ~/pd-disagg-benchmarks'
	gcloud compute tpus tpu-vm scp ./scripts/remote/*.sh \
		"$${TPU_NAME:-pd-phase0-v5e-4}:~/pd-disagg-remote/" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}"
	gcloud compute tpus tpu-vm scp ./benchmarks/*.py \
		"$${TPU_NAME:-pd-phase0-v5e-4}:~/pd-disagg-benchmarks/" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}"

phase0-bootstrap: phase0-copy
	gcloud compute tpus tpu-vm ssh "$${TPU_NAME:-pd-phase0-v5e-4}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='bash ~/pd-disagg-remote/bootstrap_phase0.sh'

phase0-verify:
	gcloud compute tpus tpu-vm ssh "$${TPU_NAME:-pd-phase0-v5e-4}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='bash ~/pd-disagg-remote/verify_phase0.sh'

phase0-bandwidth:
	gcloud compute tpus tpu-vm ssh "$${TPU_NAME:-pd-phase0-v5e-4}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='bash ~/pd-disagg-remote/run_bandwidth_phase0.sh'

phase0-serve:
	gcloud compute tpus tpu-vm ssh "$${TPU_NAME:-pd-phase0-v5e-4}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='bash ~/pd-disagg-remote/serve_phase0.sh'

phase0-smoke:
	gcloud compute tpus tpu-vm ssh "$${TPU_NAME:-pd-phase0-v5e-4}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='bash ~/pd-disagg-remote/smoke_test_phase0.sh'

phase1-copy:
	gcloud compute tpus tpu-vm ssh "$${TPU_NAME:-pd-phase0-v5e-4}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='mkdir -p ~/pd-disagg-src/pd_disagg'
	gcloud compute tpus tpu-vm scp ./src/pd_disagg/*.py \
		"$${TPU_NAME:-pd-phase0-v5e-4}:~/pd-disagg-src/pd_disagg/" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}"

phase1-smoke-load: phase1-copy
	gcloud compute tpus tpu-vm ssh "$${TPU_NAME:-pd-phase0-v5e-4}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='source /mnt/pd-disagg/venvs/vllm-tpu/bin/activate && \
			PYTHONPATH=~/pd-disagg-src python -m pd_disagg.loadgen \
			--endpoint http://127.0.0.1:8000 \
			--model Qwen/Qwen3-4B \
			--input-lengths 128,512 \
			--output-lengths 16,32 \
			--requests 6 \
			--arrival-rate 2 \
			--warmup-requests 1 \
			--output /mnt/pd-disagg/results/phase1-smoke.jsonl'

phase1-baseline: phase0-copy phase1-copy
	gcloud compute tpus tpu-vm ssh "$${TPU_NAME:-pd-phase0-v5e-4}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='bash ~/pd-disagg-remote/run_phase1_baseline.sh'

phase1-pull:
	mkdir -p artifacts/phase1
	gcloud compute tpus tpu-vm scp --recurse \
		"$${TPU_NAME:-pd-phase0-v5e-4}:/mnt/pd-disagg/results/phase1-baseline-i512-o64-r2" \
		./artifacts/phase1/ \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}"

phase1-aggregate: phase1-pull
	.venv/bin/pd-aggregate \
		--summaries artifacts/phase1/phase1-baseline-i512-o64-r2/*.summary.json \
		--output artifacts/phase1/phase1-baseline-i512-o64-r2/aggregate.json

loadgen-create:
	./scripts/cloud/create_loadgen.sh

loadgen-status:
	./scripts/cloud/status_loadgen.sh

loadgen-copy:
	gcloud compute ssh "$${LOADGEN_NAME:-pd-loadgen}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='mkdir -p ~/pd-disagg-src/pd_disagg ~/pd-disagg-driver'
	gcloud compute scp ./src/pd_disagg/*.py \
		"$${LOADGEN_NAME:-pd-loadgen}:~/pd-disagg-src/pd_disagg/" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}"
	gcloud compute scp ./scripts/driver/*.sh \
		"$${LOADGEN_NAME:-pd-loadgen}:~/pd-disagg-driver/" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}"

loadgen-bootstrap: loadgen-copy
	gcloud compute ssh "$${LOADGEN_NAME:-pd-loadgen}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='bash ~/pd-disagg-driver/bootstrap_loadgen.sh'

phase1-baseline-network: loadgen-copy
	gcloud compute ssh "$${LOADGEN_NAME:-pd-loadgen}" \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--command='TPU_ENDPOINT=http://10.128.0.2:8000 \
			bash ~/pd-disagg-driver/run_phase1_baseline.sh'

phase1-pull-network:
	mkdir -p artifacts/phase1-network
	gcloud compute scp --recurse \
		"$${LOADGEN_NAME:-pd-loadgen}:~/pd-results/phase1-baseline-i512-o64-r2" \
		./artifacts/phase1-network/ \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}"

phase1-aggregate-network: phase1-pull-network
	.venv/bin/pd-aggregate \
		--summaries artifacts/phase1-network/phase1-baseline-i512-o64-r2/*.summary.json \
		--output artifacts/phase1-network/phase1-baseline-i512-o64-r2/aggregate.json

decode-create:
	TPU_NAME=pd-decode-v5e-1 \
	QUEUED_RESOURCE_ID=pd-decode-v5e-1-q \
	DATA_DISK_NAME=pd-decode-data \
	TPU_PROVISIONING_MODEL=spot \
	./scripts/cloud/create_phase0.sh

decode-bootstrap:
	TPU_NAME=pd-decode-v5e-1 $(MAKE) phase0-bootstrap

pd-workers-start: phase0-copy
	TPU_NAME=pd-decode-v5e-1 $(MAKE) phase0-copy
	./scripts/cloud/start_pd_workers.sh

pd-router-start: loadgen-copy
	./scripts/cloud/start_pd_router.sh

pd-ici-start: phase0-copy loadgen-copy
	./scripts/cloud/start_pd_ici.sh

pd-status:
	gcloud compute tpus tpu-vm list \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--format='table(name,acceleratorType,state,health,networkEndpoints[0].ipAddress)'
	gcloud compute tpus queued-resources list \
		--project="$${PROJECT_ID:-disagg-503619}" \
		--zone="$${ZONE:-us-central1-a}" \
		--format='table(name,state.state)'
