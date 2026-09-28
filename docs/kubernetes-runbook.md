# Kubernetes and HAMi deployment plan

Status: k3s v1.35.8+k3s1 installed and node Ready on 28 September 2026.
NVIDIA RuntimeClass exists; HAMi and Kubernetes GPU workloads remain unverified.
This replaces Compose as the target GPU deployment. Local Compose remains useful
for development. The application stays on SGLang and Qwen/Qwen3-8B.
Build our own implementation from official documentation; use the course
project as inspiration, not a template to copy.

## Verified prerequisites

One Lambda H100 PCIe with 81559MiB VRAM, driver580.105.08, Docker29.2.1 and
Composev5.1.0. Release827d1d5 was uploaded. The Docker CUDA/BF16 probe PASSED:

```text
SGLang: 0.5.20
Pytorch: 2.13.0+cu130
CUDA runtime: 13.0
GPU: NVIDIA H100 PCIe
GPU memory: 79.2 GiB
PASS: CUDA allocation and BF16 matrix multiplication.
```

Kubernetes GPU access, HAMi limits and actual model inference remain unverified.

## Reproducible setup is the next implementation task

The GPU instance is temporary. Store infrastructure configuration and small
installation scripts in Git so a new host can be set up from a known commit.
The intended entrypoint is setup/bootstrap_gpu.sh; it does not exist yet.
Use separate k3s, Helm and HAMi installers, with pinned versions and appropriate
checksum verification, plus versioned cluster configuration and workload
manifests. Verify existing versions/configuration on reruns; avoid silent
reinstallation and fail clearly when prerequisites are incompatible.

Application and engine images package their dependencies. Host bootstrap installs
k3s; Helm installs HAMi into Kubernetes. Keep workload deployment a distinct step.
First source file to create is cluster/k3s/config.yaml:

```yaml
disable:
  - traefik
  - servicelb
write-kubeconfig-mode: "0600"
```

Prepare locally, commit and sync, then run on the GPU host. The manual commands
below are installation research notes, not the final reproducible interface.
Manual k3s installation has now succeeded. Capture it in a repeatable installer
that handles a matching existing cluster without unnecessary reinstallation.
Retain the working host driver.

Recreating configuration does not preserve downloaded images or model weights.
A durable cache or provider image would need separate verification and planning.
Before terminating a paid session, copy evidence to the laptop. Do not depend on
instance-local disks surviving termination.

## Deployment order

1. Docker CUDA/BF16 probe completed successfully; preserve its output.
2. Inspect NVIDIA Container Runtime and existing k3s/Helm binaries.
3. Select and pin k3s, Helm and HAMi versions; match HAMi's kube-scheduler version
   to the cluster. Reuse the installed driver and verify the NVIDIA runtime.
4. Install single-node k3s and HAMi. Keep kubeconfig private. Ensure HAMi is the
   sole NVIDIA device plugin and use verified NVIDIA runtime selection.
5. Verify GPU access and memory quota enforcement inside a Kubernetes pod.
6. Deploy two identical SGLang workers with explicit memory/compute requests.
   Recalculate engine memory settings against the HAMi-visible allocation.
7. Deploy gateway, agent app, monitoring and UI with service DNS, persistent
   caches/data and localhost access through SSH plus kubectl port-forward.
8. Verify real completions, live scrapes, application tool flow and resource
   isolation; then conduct the assignment's workload measurements.

## Choices still to verify

Exact component versions, GPU shares, SGLang internal memory budget, image import
into the k3s runtime, storage manifests and port-forward lifecycle. The Docker
image cache is not evidence that the k3s runtime already has the image. KEDA and
scaling are not included automatically by the Kubernetes/HAMi decision.

## References

- [HAMi on k3s](https://project-hami.io/docs/installation/k3s-installation)
- [SGLang on HAMi GPU shares](https://project-hami.io/tutorials/labs/hami-sglang)

The user performs infrastructure commands. No installation is implied by this
plan. At session end, save evidence and terminate the cloud instance through
Lambda; stopping containers alone does not stop instance billing.

## Manual k3s commands - reference for the installer implementation

Prerequisite output confirms NVIDIA Container Runtime1.18.1 and runc1.3.4.
k3s and Helm were not found on PATH. These commands were proposed before the
user requested repository-managed bootstrap. Incorporate them into the reviewed
installer. User executed these commands successfully; node Ready and RuntimeClass
nvidia were observed. The commands alone do not verify GPU workloads or HAMi.

```bash
curl -fL 'https://raw.githubusercontent.com/k3s-io/k3s/v1.35.8+k3s1/install.sh' -o /tmp/course-k3s-install.sh
```

```bash
sudo env INSTALL_K3S_VERSION='v1.35.8+k3s1' sh /tmp/course-k3s-install.sh server --disable=traefik --disable=servicelb --write-kubeconfig-mode=600
```

```bash
sudo k3s kubectl get nodes -o wide
```

```bash
sudo k3s kubectl get runtimeclass nvidia
```

The server uses embedded containerd. Traefik and ServiceLB are disabled because
this lab uses SSH and port-forward access. Kubeconfig stays mode0600. The NVIDIA
runtime will be explicitly selected on HAMi device-plugin and GPU workload pods;
RuntimeClass existence alone does not establish correct runtime configuration.

Sources: [versioned release](https://github.com/k3s-io/k3s/releases/tag/v1.35.8%2Bk3s1),
[NVIDIA runtime configuration](https://docs.k3s.io/advanced#nvidia-container-runtime).

## Observed cluster installation result

User output: node Ready, k3s v1.35.8+k3s1, Ubuntu22.04.5 LTS,
kernel6.8.0-1046-nvidia, containerd2.2.7-k3s1, RuntimeClass nvidia with handler
nvidia. The installer verified the downloaded binary hash. No HAMi installation
or GPU pod execution has been reported. A generic ctr binary was already on PATH;
use sudo k3s ctr when interacting with the k3s image store.

## Local installer review checkpoint

The student created cluster/k3s/config.yaml and setup/install_k3s.sh. Shell syntax
passes. Four isolated mocked shell checks pass: retry until Ready, deadline
failure, rejection of an incompatible installed k3s version, and reusing an
installed binary without downloading again. These do not prove fresh-host
installation or Kubernetes GPU access. The installer has not run remotely yet.

Next: commit the installer, config and runbooks, sync the new commit, reconnect
into that release and run bash setup/install_k3s.sh on the GPU host. It should
reuse the already-installed matching k3s version. Inspect the real result before
marking rerun behavior verified. Helm/HAMi and full bootstrap orchestration are
still to be implemented. The current paid instance may be terminated by the user;
confirm host status before resuming remote work.
