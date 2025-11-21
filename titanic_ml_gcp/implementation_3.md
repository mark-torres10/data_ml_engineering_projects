# Phase 3: Kubernetes Deployment with GKE Autopilot

## Overview

This phase transforms your ML application into a production-ready system running on Google Kubernetes Engine (GKE) Autopilot. You'll learn container orchestration, service mesh concepts, auto-scaling, and cloud-native monitoring.

**Prerequisites:** Complete Phase 1 and Phase 2 successfully

**Time Estimate:** 2-3 days

**Cost Estimate:** $10-15 (GKE cluster and load balancer costs)

---

## Step 1: Understanding Kubernetes Concepts

### 1.1 What is Kubernetes?

**Purpose:**
- Container orchestration platform
- Automates deployment, scaling, and management of containerized applications
- Provides service discovery, load balancing, and self-healing
- Declarative configuration (infrastructure as code)

**Key benefits:**
- High availability (automatic restart of failed containers)
- Horizontal scaling (add/remove replicas based on load)
- Rolling updates (deploy new versions without downtime)
- Resource optimization (efficient use of compute resources)
- Portable across clouds (works on GCP, AWS, Azure, on-prem)

**AWS Comparison:** GKE is equivalent to Amazon EKS (Elastic Kubernetes Service).

### 1.2 Core Kubernetes Concepts

**Cluster:**
- Collection of nodes (machines) running Kubernetes
- Control plane: Manages the cluster
- Worker nodes: Run your applications

**Pod:**
- Smallest deployable unit
- One or more containers that share storage and network
- Usually one container per pod
- Ephemeral: Can be created/destroyed frequently

**Deployment:**
- Declares desired state for pods
- Manages replica sets
- Handles rolling updates and rollbacks
- Ensures specified number of pods are running

**Service:**
- Stable endpoint for pods
- Load balances traffic across pod replicas
- Types: ClusterIP (internal), LoadBalancer (external), NodePort

**ConfigMap:**
- Store configuration data as key-value pairs
- Decouple configuration from container images
- Can be mounted as files or environment variables

**Secret:**
- Like ConfigMap but for sensitive data
- Stored encrypted in etcd
- Used for API keys, passwords, certificates

**Ingress:**
- Manages external HTTP/HTTPS access to services
- Provides load balancing, SSL termination, routing
- Single entry point for multiple services

**Namespace:**
- Virtual clusters within physical cluster
- Isolate resources between teams or environments
- Default namespaces: default, kube-system, kube-public

### 1.3 GKE Autopilot vs Standard

**GKE Standard:**
- You manage nodes (machine types, updates, security)
- Full control over cluster configuration
- More flexibility but more operational overhead
- Pay for provisioned nodes even if unused

**GKE Autopilot (Recommended for this project):**
- Google manages nodes, updates, security
- You only configure pods and workloads
- Automatic scaling and security hardening
- Pay only for pods running (per-pod pricing)
- Opinionated best practices enforced

**For learning:** Autopilot is ideal - less infrastructure management, more focus on application.

---

## Step 2: Prerequisites and Setup

### 2.1 Install kubectl

**What is kubectl:**
- Command-line tool for Kubernetes
- Used to deploy applications, inspect resources, view logs
- Essential for Kubernetes operations

**Installation:**
- On macOS: Use Homebrew
- On Linux: Download binary from Kubernetes releases
- On Windows: Use Chocolatey or download binary
- Verify installation with `kubectl version --client`

**Configuration:**
- kubectl uses kubeconfig file (usually ~/.kube/config)
- Stores cluster connection details and credentials
- Can manage multiple clusters

**Basic commands to learn:**
- kubectl get: List resources
- kubectl describe: Show detailed info
- kubectl logs: View container logs
- kubectl apply: Apply configuration from file
- kubectl delete: Delete resources
- kubectl exec: Execute command in container

### 2.2 Install Docker

**Verify Docker installation:**
- Should already be installed from Phase 1
- Check with `docker --version`
- Ensure Docker daemon is running

**Docker configuration:**
- Configure Docker with adequate resources (4GB+ memory)
- Enable Kubernetes in Docker Desktop (optional, for local testing)

### 2.3 Set Up gcloud for GKE

**Install GKE components:**
- Run `gcloud components install gke-gcloud-auth-plugin`
- Required for authenticating kubectl with GKE

**Configure gcloud:**
- Ensure default project is set correctly
- Ensure default region is set (e.g., us-central1)
- Verify authentication is active

---

## Step 3: Containerize the Streamlit Application

### 3.1 Create Dockerfile for Streamlit

**Create deployment/docker/Dockerfile.streamlit:**

**Base image selection:**
- Use official Python image (python:3.12-slim)
- Slim variant reduces image size
- Includes minimal system dependencies
- Python 3.12 matches local development environment

**Dockerfile structure:**

**Set working directory:**
- Create /app directory
- All subsequent commands run from here

**Copy requirements:**
- Copy requirements.txt first (before code)
- Enables Docker layer caching
- Pip install only runs when requirements change

**Install dependencies:**
- Use uv for significantly faster builds: `COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv`
- Run `uv pip install --no-cache` to reduce image size
- Alternative: Use traditional pip install with --no-cache-dir
- Pin all versions for reproducibility
- Consider multi-stage build for smaller final image
- uv can reduce Docker build times from 5 minutes to under 1 minute

**Copy application code:**
- Copy src/ directory
- Copy any configuration files
- Exclude unnecessary files (.gitignore, tests, notebooks)

**Set environment variables:**
- STREAMLIT_SERVER_PORT=8080 (GCP standard)
- STREAMLIT_SERVER_ADDRESS=0.0.0.0 (listen on all interfaces)
- Any GCP project or bucket names

**Expose port:**
- EXPOSE 8080
- Documents which port the container listens on
- Required by Kubernetes Service

**Set entrypoint:**
- Run streamlit with --server.port=8080
- Point to your main app file
- Use --server.headless=true for containerized environment

### 3.2 Optimize Docker Image

**Multi-stage build (optional but recommended):**
- First stage: Build dependencies and prepare environment
- Second stage: Copy only necessary artifacts
- Results in smaller final image

**Reduce layers:**
- Combine RUN commands with && where possible
- Clean up package manager caches in same layer
- Remove temporary files before layer commits

**Minimize image size:**
- Use .dockerignore to exclude files
- Don't include .git, __pycache__, notebooks, tests
- Remove build tools not needed at runtime

**Security considerations:**
- Don't run as root (create non-root user)
- Use specific version tags, not :latest
- Scan images for vulnerabilities (gcloud artifacts docker images scan)

### 3.3 Test Dockerfile Locally

**Build image:**
- Run docker build command with appropriate tag
- Example tag: gcr.io/PROJECT_ID/titanic-streamlit:v1
- Watch for errors or warnings during build
- Build typically takes 2-5 minutes

**Run container locally:**
- Use docker run with port mapping (-p 8080:8080)
- Mount any required local volumes for testing
- Set environment variables with -e flags
- Test that application starts correctly

**Verify functionality:**
- Access http://localhost:8080 in browser
- Test all UI features work
- Test endpoint connectivity (may need to adjust host references)
- Check logs with docker logs command

**Common issues:**
- Port already in use: Choose different host port
- Cannot connect to endpoint: Update endpoint URL for container context
- Missing files: Check COPY commands in Dockerfile
- Permission errors: Ensure proper file permissions

### 3.4 Push Image to Artifact Registry

**Create repository in Artifact Registry:**
- Navigate to Artifact Registry in GCP Console
- Click "Create Repository"
- Name: titanic-ml-images
- Format: Docker
- Region: Same as your GKE cluster (e.g., us-central1)
- Encryption: Google-managed key

**Authenticate Docker with Artifact Registry:**
- Run gcloud auth configure-docker command with your region
- Example: `gcloud auth configure-docker us-central1-docker.pkg.dev`
- Adds credential helper to Docker config

**Tag image with registry path:**
- Format: REGION-docker.pkg.dev/PROJECT_ID/REPO_NAME/IMAGE:TAG
- Example: us-central1-docker.pkg.dev/my-project/titanic-ml-images/streamlit:v1
- Use docker tag command to rename existing image

**Push image:**
- Run docker push with full image path
- Upload takes 2-5 minutes depending on image size and connection
- Verify in Artifact Registry console

**Image tagging strategy:**
- Use semantic versions (v1.0.0, v1.1.0)
- Use git commit SHA for traceability
- Tag production deployments as 'prod' or 'latest'
- Never reuse tags (immutable tags)

---

## Step 4: Create Inference Service Container (Optional)

### 4.1 Decide on Architecture

**Two deployment options:**

**Option A: Streamlit calls Vertex AI endpoint directly (Simpler)**
- Streamlit container makes API calls to Vertex AI
- No separate inference service needed
- Fewer containers to manage
- Good for starting out

**Option B: Separate inference service (More flexible)**
- Streamlit calls your own inference service
- Inference service loads model from GCS
- Can switch models without Vertex AI redeployment
- Better for cost optimization (no Vertex AI endpoint charges)
- More complex to set up

**For this project:** Start with Option A, can migrate to Option B later if desired.

### 4.2 Create Inference Service (if using Option B)

**Create deployment/docker/Dockerfile.inference:**

**Similar structure to Streamlit Dockerfile:**
- Python base image
- Install FastAPI or Flask
- Install ML dependencies (xgboost, scikit-learn, numpy)
- Copy prediction code
- Copy or download model from GCS
- Expose port 8080

**Create src/models/inference_server.py:**

**API endpoints to implement:**
- GET /health - Health check (returns 200 OK)
- POST /predict - Main prediction endpoint
- GET /model/info - Return model version and metadata
- POST /explain - Return predictions with explanations

**Request/response format:**
- Accept JSON with passenger features
- Return JSON with prediction and probability
- Include model version in response
- Handle errors gracefully

**Model loading:**
- Load model from GCS on startup
- Cache in memory for fast predictions
- Reload when new version detected
- Implement graceful degradation if load fails

**Test locally:**
- Build Docker image
- Run container
- Test with curl or Postman
- Verify predictions match expected results

**Push to Artifact Registry:**
- Tag as inference:v1
- Push to same repository

---

## Step 5: Create GKE Autopilot Cluster

### 5.1 Plan Cluster Configuration

**Cluster decisions:**
- Name: titanic-ml-cluster
- Region: us-central1 (or your preferred region)
- Autopilot mode: Enabled
- Release channel: Regular (balanced stability and features)
- Version: Latest stable Kubernetes version

**Networking:**
- VPC: Default or create new VPC
- Subnet: Automatically managed by Autopilot
- IP ranges: Default (Autopilot handles)
- Network policy: Enabled (for security)

**Security:**
- Workload Identity: Enable (recommended for GCP service authentication)
- Binary Authorization: Optional (for signed container images)
- Shielded nodes: Enabled by default in Autopilot

### 5.2 Create the Cluster

**Using gcloud CLI (Recommended):**
- Run gcloud container clusters create-auto command
- Specify cluster name and region
- Enable Workload Identity
- Creation takes 5-10 minutes

**Using GCP Console:**
- Navigate to Kubernetes Engine > Clusters
- Click "Create"
- Select "Autopilot" mode
- Configure name and region
- Accept default settings
- Click "Create"

**Cluster creation process:**
- Provisions control plane
- Sets up node pools (managed by Autopilot)
- Configures networking
- Applies security policies

**Verify cluster:**
- Check cluster status in console
- Cluster shows as "Running" when ready

### 5.3 Configure kubectl

**Get cluster credentials:**
- Run gcloud container clusters get-credentials command
- Specifies cluster name and region
- Updates ~/.kube/config with cluster info
- Sets as current context

**Verify connection:**
- Run `kubectl get nodes` to see nodes (may show none in Autopilot until pods scheduled)
- Run `kubectl cluster-info` to see cluster endpoints
- Run `kubectl get namespaces` to see default namespaces

**Understand Autopilot behavior:**
- Nodes auto-provision when pods are scheduled
- Nodes auto-scale based on pod resource requests
- Nodes are abstracted - you don't manage them directly
- You only see pods, not individual nodes (in Autopilot)

---

## Step 6: Create Kubernetes Manifests

### 6.1 Create Namespace

**Purpose:** Isolate resources for this application

**Create deployment/kubernetes/namespace.yaml:**
- Define namespace named "titanic-ml"
- Add labels for organization
- Include annotations if desired

**Apply namespace:**
- Use kubectl apply -f namespace.yaml
- Verify with kubectl get namespaces
- Set as default context to avoid specifying namespace repeatedly

### 6.2 Create ConfigMap

**Purpose:** Store non-sensitive configuration

**Create deployment/kubernetes/configmap.yaml:**

**Configuration to store:**
- GCP project ID
- GCS bucket names
- Vertex AI endpoint ID
- MLflow tracking URI
- Model version to use
- Application settings

**ConfigMap structure:**
- Name: titanic-ml-config
- Namespace: titanic-ml
- Data section with key-value pairs

**Apply ConfigMap:**
- Use kubectl apply
- Verify with kubectl get configmap

### 6.3 Create Secret

**Purpose:** Store sensitive data securely

**Create deployment/kubernetes/secret.yaml:**

**Secrets to store:**
- GCP service account key (JSON)
- API keys if any
- Database passwords if any

**Secret types:**
- Opaque: Generic secret
- kubernetes.io/service-account-token: For service accounts
- kubernetes.io/dockerconfigjson: For registry authentication

**Create secret:**
- Base64 encode sensitive values
- Include in secret manifest
- Or create via kubectl create secret command

**Security notes:**
- Never commit secrets to git
- Use external secret management (GCP Secret Manager) for production
- Rotate secrets regularly
- Minimize secrets stored in Kubernetes

### 6.4 Set Up Workload Identity (Recommended)

**What is Workload Identity:**
- Maps Kubernetes service accounts to GCP service accounts
- Eliminates need for JSON keys in containers
- More secure than using service account keys
- GCP-recommended authentication method

**Enable for cluster:**
- Workload Identity enabled during cluster creation
- Or enable on existing cluster

**Create GCP service account:**
- Create service account for application
- Grant necessary IAM roles (Vertex AI User, Storage Object Viewer)

**Create Kubernetes service account:**
- Define in manifest
- Annotate with GCP service account email

**Bind accounts:**
- Use gcloud iam service-accounts add-iam-policy-binding
- Allows Kubernetes SA to impersonate GCP SA

**Configure pod:**
- Specify serviceAccountName in pod spec
- Application automatically gets GCP credentials

### 6.5 Create Deployment Manifest

**Create deployment/kubernetes/deployment.yaml:**

**Deployment specification:**

**Metadata:**
- Name: titanic-streamlit
- Namespace: titanic-ml
- Labels: app=titanic, component=frontend

**Replica count:**
- Start with replicas: 2
- Provides redundancy
- Can be scaled later

**Pod template:**

**Containers:**
- Name: streamlit
- Image: Full path to Artifact Registry image
- ImagePullPolicy: Always (pulls latest) or IfNotPresent
- Ports: containerPort 8080

**Resource requests (required in Autopilot):**
- CPU: 500m (millicores)
- Memory: 1Gi
- These determine node provisioning

**Resource limits:**
- CPU: 1000m
- Memory: 2Gi
- Prevents runaway resource consumption

**Environment variables:**
- Reference ConfigMap values
- Reference Secret values
- Set any additional runtime variables

**Volume mounts:**
- Mount ConfigMaps as files if needed
- Mount Secrets as files
- Typically not needed if using env vars

**Liveness probe:**
- Checks if container is alive
- HTTP GET to /health or similar endpoint
- Restart container if probe fails
- initialDelaySeconds: 30 (wait for startup)
- periodSeconds: 10

**Readiness probe:**
- Checks if container is ready to serve traffic
- Similar to liveness but different purpose
- Don't route traffic until ready
- initialDelaySeconds: 15
- periodSeconds: 5

**Security context:**
- Run as non-root user
- Set readOnlyRootFilesystem: true if possible
- Drop unnecessary capabilities

### 6.6 Create Service Manifest

**Create deployment/kubernetes/service.yaml:**

**Service type: LoadBalancer**
- Creates external load balancer
- Provides stable external IP
- Routes traffic to pods

**Service specification:**
- Name: titanic-streamlit-service
- Namespace: titanic-ml
- Type: LoadBalancer
- Selector: app=titanic, component=frontend (matches deployment labels)

**Ports:**
- Protocol: TCP
- Port: 80 (external port users access)
- TargetPort: 8080 (container port)
- Name: http

**Alternative: ClusterIP with Ingress**
- Service type: ClusterIP (internal only)
- Create Ingress resource for external access
- Provides more features (SSL, routing, etc.)

### 6.7 Create Ingress (Optional but Recommended)

**Purpose:** Better external access control

**Create deployment/kubernetes/ingress.yaml:**

**Ingress features:**
- Single external IP for multiple services
- Path-based routing (/api, /ui, etc.)
- Host-based routing (different domains)
- SSL/TLS termination
- Google Cloud Load Balancer integration

**Ingress specification:**
- IngressClass: gce (Google Cloud Load Balancer)
- Rules: Define routing logic
- Backend: Point to Service
- TLS: Configure SSL certificates if desired

**SSL certificate:**
- Use Google-managed certificate
- Or use cert-manager with Let's Encrypt
- Or upload your own certificate

---

## Step 7: Deploy Application to GKE

### 7.1 Apply Kubernetes Manifests

**Deployment order:**
- Namespace first
- ConfigMap and Secret next
- Deployment third
- Service last
- Ingress if using

**Apply commands:**
- Use kubectl apply -f for each file
- Or apply entire directory: kubectl apply -f deployment/kubernetes/
- Kubernetes applies in correct dependency order

**Watch deployment progress:**
- Use kubectl get pods -n titanic-ml -w (watch mode)
- Pods go through: Pending -> ContainerCreating -> Running
- In Autopilot, node provisioning adds extra time initially

### 7.2 Verify Deployment

**Check pod status:**
- kubectl get pods -n titanic-ml
- All pods should show STATUS: Running
- READY should show 1/1 or expected count

**Check pod details:**
- kubectl describe pod POD_NAME -n titanic-ml
- Shows events, mounts, conditions
- Check for error messages

**Check logs:**
- kubectl logs POD_NAME -n titanic-ml
- Should see Streamlit startup messages
- No errors about missing config or credentials

**Check service:**
- kubectl get service -n titanic-ml
- LoadBalancer should show EXTERNAL-IP (takes 2-3 minutes)
- Note the external IP address

### 7.3 Access the Application

**Get external IP:**
- From kubectl get service output
- Or from GCP Console > Kubernetes Engine > Services

**Access in browser:**
- Navigate to http://EXTERNAL_IP
- Application should load
- Test all functionality

**Test predictions:**
- Input passenger details
- Verify predictions work
- Check that Vertex AI endpoint is reachable
- Verify explanations display

### 7.4 Troubleshooting Deployment Issues

**Pod stuck in Pending:**
- Check resource quotas (unlikely in Autopilot)
- Check resource requests are reasonable
- View events: kubectl describe pod

**Pod CrashLoopBackOff:**
- Application crashing on startup
- Check logs: kubectl logs POD_NAME
- Common causes: Missing dependencies, bad configuration, no liveness probe delay

**ImagePullBackOff:**
- Cannot pull image from registry
- Verify image path is correct
- Check Workload Identity or image pull secrets
- Ensure repository permissions

**Service no external IP:**
- LoadBalancer creation takes time (wait 5 minutes)
- Check GCP quotas for load balancers
- Verify service type is LoadBalancer
- Check service events: kubectl describe service

**Cannot connect to endpoint:**
- Verify GOOGLE_APPLICATION_CREDENTIALS or Workload Identity
- Check service account permissions
- Ensure endpoint ID is correct in ConfigMap
- Test endpoint connectivity from pod: kubectl exec

---

## Step 8: Implement Auto-Scaling

### 8.1 Horizontal Pod Autoscaler (HPA)

**What is HPA:**
- Automatically scales number of pod replicas
- Based on observed CPU, memory, or custom metrics
- Increases pods when load is high
- Decreases pods when load is low

**Create HPA manifest:**

**Create deployment/kubernetes/hpa.yaml:**

**HPA specification:**
- ScaleTargetRef: Points to your Deployment
- MinReplicas: 2 (minimum pods always running)
- MaxReplicas: 10 (maximum pods for burst capacity)

**Metrics:**
- CPU: Target 70% average utilization
- Memory: Target 80% average utilization

**Behavior:**
- ScaleUp: How quickly to add pods
- ScaleDown: How slowly to remove pods (avoid thrashing)

**Apply HPA:**
- kubectl apply -f hpa.yaml
- Verify: kubectl get hpa -n titanic-ml

### 8.2 Vertical Pod Autoscaler (VPA) (Optional)

**What is VPA:**
- Adjusts CPU and memory requests/limits automatically
- Based on actual usage patterns
- Helps right-size resource allocations
- Typically used in "recommendation mode" to inform manual adjustments

**When to use:**
- Not immediately necessary
- Useful after observing actual usage
- Can conflict with HPA if both active

### 8.3 Test Auto-Scaling

**Generate load:**
- Use load testing tool (Apache Bench, Locust, hey)
- Send many concurrent requests
- Monitor pod count with: kubectl get pods -w

**Observe scaling:**
- HPA should detect high CPU/memory
- New pods are created (takes 1-2 minutes)
- Load balancer distributes traffic
- When load decreases, pods are terminated (after cooldown period)

**Verify scaling behavior:**
- Check HPA status: kubectl describe hpa
- Shows current vs target metrics
- Shows scaling events

**Tune scaling parameters:**
- Adjust target utilization based on observations
- Adjust min/max replicas
- Adjust scaling policies (speed of scale up/down)

---

## Step 9: Set Up Monitoring and Logging

### 9.1 Google Cloud Logging Integration

**Automatic logging:**
- GKE automatically sends container logs to Cloud Logging
- Includes stdout and stderr from containers
- Structured logging if application outputs JSON

**View logs:**
- GCP Console > Logging > Logs Explorer
- Filter by resource.type="k8s_container"
- Filter by namespace_name="titanic-ml"
- Filter by pod_name, container_name as needed

**Create log-based metrics:**
- Count error messages
- Track prediction requests
- Monitor model loading events
- Create custom metrics from log patterns

**Set up log-based alerts:**
- Alert on high error rate
- Alert on application crashes
- Alert on failed predictions

### 9.2 Google Cloud Monitoring Integration

**Automatic metrics:**
- GKE sends metrics to Cloud Monitoring
- Pod CPU and memory usage
- Container restarts
- Network traffic

**View metrics:**
- GCP Console > Monitoring > Metrics Explorer
- Select Kubernetes resource types
- Choose metrics to plot
- Create custom dashboards

**Key metrics to monitor:**
- Pod CPU utilization
- Pod memory utilization
- Request rate (from Streamlit)
- Request latency
- Error rate
- Prediction endpoint latency

### 9.3 Create Custom Dashboard

**Dashboard creation:**
- Navigate to Cloud Monitoring > Dashboards
- Create new dashboard: "Titanic ML Application"

**Widgets to add:**

**Pod health:**
- Number of running pods
- Pod restart count
- Pod creation/deletion events

**Resource usage:**
- CPU utilization across all pods
- Memory utilization across all pods
- Network I/O

**Application metrics:**
- Request rate (if instrumented)
- Request latency (if instrumented)
- Error rate (from logs)

**ML-specific metrics:**
- Prediction requests per minute
- Average confidence scores
- Model version in use

**Layout:**
- Arrange logically (health at top, details below)
- Use appropriate chart types (line, gauge, table)
- Set time range to Last 1 hour or Last 6 hours

### 9.4 Set Up Alerting

**Create alert policies:**

**Pod availability alert:**
- Condition: Number of running pods < desired count
- Duration: 5 minutes
- Notification: Email, Slack, PagerDuty

**High CPU alert:**
- Condition: Pod CPU > 90% for 10 minutes
- Indicates need to scale or investigate
- May indicate infinite loop or resource leak

**High memory alert:**
- Condition: Pod memory > 90% for 10 minutes
- Risk of OOMKilled
- May indicate memory leak

**Error rate alert:**
- Condition: Error logs > 10 per minute
- Indicates application issues
- Requires investigation

**Endpoint latency alert:**
- Condition: Prediction latency > 1 second
- Degraded user experience
- May indicate endpoint or network issues

**Notification channels:**
- Email: For non-critical alerts
- SMS: For urgent issues
- Slack/PagerDuty: For team collaboration
- Webhook: For custom integrations

---

## Step 10: Implement CI/CD (Optional but Recommended)

### 10.1 Overview of CI/CD for Kubernetes

**Continuous Integration:**
- Automatic build and test on code commit
- Build Docker images automatically
- Run tests in containerized environment
- Push images to Artifact Registry

**Continuous Deployment:**
- Automatically deploy to Kubernetes
- Update deployments with new images
- Rolling update strategy
- Rollback on failure

**Tools:**
- Cloud Build (GCP native)
- GitHub Actions
- GitLab CI
- Jenkins

### 10.2 Set Up Cloud Build

**Create cloudbuild.yaml:**

**Build steps:**

**Step 1: Build Docker image**
- Use docker build command
- Tag with commit SHA and branch name
- Use Cloud Build's native Docker builder

**Step 2: Push to Artifact Registry**
- Push image with multiple tags
- Latest tag for quick access
- Version tag for tracking

**Step 3: Update Kubernetes deployment**
- Use kubectl set image command
- Or use kustomize for more control
- Update deployment to use new image

**Step 4: Verify deployment**
- Check rollout status
- Run smoke tests
- Rollback if tests fail

**Trigger setup:**
- Trigger on push to main branch
- Trigger on tag push for releases
- Manual trigger for production

### 10.3 Implement GitOps (Advanced)

**What is GitOps:**
- Git as single source of truth
- Declarative infrastructure and applications
- Automated deployment on Git changes
- Easy rollback via Git

**Tools:**
- Argo CD
- Flux CD
- Config Sync (GCP native)

**Workflow:**
- Commit Kubernetes manifests to Git
- GitOps tool monitors repo
- Automatically applies changes to cluster
- Ensures cluster state matches Git

---

## Step 11: Security Best Practices

### 11.1 Pod Security

**Run as non-root:**
- Set securityContext.runAsNonRoot: true
- Set securityContext.runAsUser: 1000
- Prevents privilege escalation

**Read-only root filesystem:**
- Set securityContext.readOnlyRootFilesystem: true
- Use volume mounts for writable directories
- Reduces attack surface

**Drop capabilities:**
- Drop ALL capabilities
- Add back only what's needed
- Minimize container permissions

**Pod Security Standards:**
- Kubernetes defines three levels: Privileged, Baseline, Restricted
- Aim for Restricted where possible
- Enforced via Pod Security Admission

### 11.2 Network Security

**Network Policies:**
- Define which pods can communicate
- Default deny all traffic
- Explicitly allow necessary traffic

**Example policy:**
- Allow Streamlit pods to access internet (for Vertex AI)
- Allow Streamlit pods to communicate with inference service
- Deny all other traffic

**Service mesh (Advanced):**
- Istio or Anthos Service Mesh
- Provides mutual TLS between services
- Fine-grained authorization policies
- Not necessary for simple applications

### 11.3 Secrets Management

**GCP Secret Manager integration:**
- Store secrets in Secret Manager, not Kubernetes
- Use External Secrets Operator to sync to Kubernetes
- Centralized secret management
- Audit logging of secret access

**Workload Identity:**
- Always use instead of service account keys
- No long-lived credentials in cluster
- Automatic credential rotation

### 11.4 Image Security

**Scan images for vulnerabilities:**
- Enable vulnerability scanning in Artifact Registry
- Review scan results before deployment
- Address critical and high vulnerabilities

**Use minimal base images:**
- Alpine or distroless images
- Fewer packages = smaller attack surface
- Regular updates

**Sign images:**
- Binary Authorization (GCP)
- Only deploy signed images
- Ensures images haven't been tampered with

---

## Step 12: Cost Optimization

### 12.1 Optimize Resource Requests

**Right-size pods:**
- Monitor actual CPU and memory usage
- Adjust requests to match reality
- Don't over-provision (wastes money)
- Don't under-provision (causes instability)

**Use VPA recommendations:**
- Run VPA in recommendation mode
- Review suggested requests
- Apply manually with testing

### 12.2 Optimize Scaling

**Set appropriate min/max replicas:**
- Min: Enough for availability (2-3)
- Max: Budget constraint or expected peak
- Don't set max too high without testing costs

**Tune scaling thresholds:**
- Higher CPU target = fewer pods = lower cost
- But may impact performance
- Balance cost vs performance

**Scale to zero (if acceptable):**
- Some workloads can scale to 0 when idle
- Use Cloud Run instead of GKE for this use case
- Not typical for user-facing applications

### 12.3 Use Spot/Preemptible Nodes (Standard GKE)

**Not applicable to Autopilot:**
- Autopilot manages nodes automatically
- If using Standard GKE, consider spot nodes for dev/test

### 12.4 Monitor Costs

**Set up budget alerts:**
- GCP Console > Billing > Budgets & Alerts
- Set monthly budget
- Alert at 50%, 90%, 100%

**Review costs regularly:**
- Check Cloud Billing reports
- Identify most expensive resources
- Look for optimization opportunities

**Clean up unused resources:**
- Delete test clusters
- Remove old load balancers
- Archive old container images

---

## Step 13: Testing and Validation

### 13.1 Functional Testing

**Test all application features:**
- Predictions work correctly
- Model retraining works
- Metrics display properly
- Explanations render

**Test across pod restarts:**
- Delete a pod, verify new one starts
- Ensure no data loss
- Verify connections re-establish

**Test during scaling events:**
- Generate load, verify new pods serve correctly
- Reduce load, verify scale-down doesn't break anything

### 13.2 Load Testing

**Tools:**
- Locust: Python-based, flexible
- Apache Bench: Simple, command-line
- hey: Modern alternative to Apache Bench
- Artillery: Node.js-based

**Load test scenarios:**
- Gradual ramp-up: 0 to 100 users over 5 minutes
- Sustained load: 100 concurrent users for 30 minutes
- Spike: Sudden jump from 10 to 200 users
- Stress test: Push beyond expected maximum

**Metrics to observe:**
- Request success rate (should be >99%)
- Request latency (p50, p95, p99)
- Error rate (should be <1%)
- Pod scaling behavior
- Resource utilization

### 13.3 Chaos Engineering (Optional)

**Test resilience:**
- Delete random pods (kubectl delete pod)
- Introduce network latency
- Simulate partial outages
- Verify application recovers gracefully

**Tools:**
- Chaos Mesh
- Litmus Chaos
- Gremlin (commercial)

### 13.4 Disaster Recovery

**Test backup and restore:**
- Backup Kubernetes manifests (they're in Git, right?)
- Backup critical data in GCS
- Test restoring in new cluster
- Document recovery procedures

**Recovery Time Objective (RTO):**
- How long to restore service
- For this project: Manual restore acceptable
- For production: Automated failover

**Recovery Point Objective (RPO):**
- How much data loss is acceptable
- Models and data in GCS are durable
- Application state should be stateless

---

## Step 14: Documentation

### 14.1 Architecture Documentation

**Create architecture diagram:**
- Show all components (GKE, Artifact Registry, GCS, Vertex AI)
- Show data flow
- Show network boundaries
- Include both user flow and training flow

**Document design decisions:**
- Why Autopilot vs Standard
- Why LoadBalancer vs Ingress
- Resource sizing rationale
- Scaling parameters chosen

### 14.2 Operations Runbook

**Common operations:**

**Deploy new version:**
- Build and push image
- Update deployment manifest
- Apply with kubectl
- Verify rollout

**Scale manually:**
- Edit HPA manifest or use kubectl scale
- When and why to scale

**View logs:**
- Commands to view pod logs
- How to filter in Cloud Logging
- What to look for

**Troubleshoot issues:**
- Common problems and solutions
- Where to check first (logs, metrics, events)
- Escalation procedures

**Rollback:**
- How to rollback deployment
- Use kubectl rollout undo
- When to rollback vs fix forward

### 14.3 Developer Guide

**Local development:**
- How to test Dockerfile locally
- How to test Kubernetes manifests with kind or minikube
- How to debug issues

**Making changes:**
- Where to edit code
- How to build and push images
- How to update manifests
- How to test changes

**Adding features:**
- How to add new environment variables
- How to add new dependencies
- How to expose new endpoints

---

## Step 15: Cleanup and Cost Management

### 15.1 Temporary Shutdown

**When pausing work:**

**Stop workloads:**
- Scale deployment to 0 replicas
- Or delete deployment entirely
- Keeps cluster but stops pod charges

**Cluster options:**
- Delete cluster to stop all charges
- Or keep cluster (control plane has minimal cost)
- Autopilot charges per pod, so 0 pods = minimal cost

**What to keep:**
- Container images in Artifact Registry (storage is cheap)
- Models and data in GCS
- Git repository with all manifests

### 15.2 Complete Cleanup

**Delete all resources:**

**Delete Kubernetes resources:**
- kubectl delete namespace titanic-ml (deletes everything in namespace)
- Or delete cluster entirely

**Delete GKE cluster:**
- gcloud container clusters delete command
- Or use GCP Console
- Deletes associated load balancers automatically

**Delete container images (optional):**
- Artifact Registry > select images > delete
- Or keep for future use (minimal cost)

**Delete other resources:**
- Vertex AI endpoints (if still running)
- GCS buckets (if no longer needed)
- Service accounts (if no longer needed)

**Verify no lingering resources:**
- Check GCP Console for orphaned resources
- Load balancers sometimes remain (delete manually)
- Persistent disks (should be none for this project)

### 15.3 Cost Optimization for Ongoing Use

**For continued learning/demo:**
- Scale to 1 replica minimum
- Use smaller machine types if load is low
- Delete endpoint when not actively demoing
- Set aggressive scale-down policies

**For actual production use:**
- Right-size based on actual usage
- Use committed use discounts for predictable workloads
- Consider Cloud Run for variable/bursty workloads
- Monitor and optimize continuously

---

## Troubleshooting Common Issues

### Issue: Pods stuck in ImagePullBackOff

**Causes:**
- Incorrect image path
- Missing registry authentication
- Image doesn't exist

**Solutions:**
- Verify image path exactly matches Artifact Registry
- Check Workload Identity is configured
- Verify image was pushed successfully
- Test pull locally: docker pull IMAGE_PATH

### Issue: Service has no external IP

**Causes:**
- LoadBalancer creation in progress
- Quota exceeded
- Region doesn't support LoadBalancer

**Solutions:**
- Wait 5-10 minutes
- Check GCP quotas for load balancers
- Verify service type is LoadBalancer
- Check service events: kubectl describe service

### Issue: Application can't reach Vertex AI endpoint

**Causes:**
- Missing authentication
- Workload Identity not configured
- Service account lacks permissions
- Incorrect endpoint ID

**Solutions:**
- Verify Workload Identity setup
- Check service account has Vertex AI User role
- Test from pod: kubectl exec -it POD -- env
- Verify GOOGLE_APPLICATION_CREDENTIALS or Workload Identity annotation

### Issue: High latency or timeouts

**Causes:**
- Insufficient resources
- Cold starts
- Network issues
- Vertex AI endpoint issues

**Solutions:**
- Increase resource requests/limits
- Increase min replicas to avoid cold starts
- Check network connectivity from pod
- Test endpoint latency separately

### Issue: Deployment rollout stuck

**Causes:**
- New pods failing health checks
- Insufficient resources
- ImagePullBackOff

**Solutions:**
- Check pod status: kubectl get pods
- Check pod events: kubectl describe pod
- Check logs: kubectl logs POD_NAME
- Fix underlying issue and reapply

---

## Success Criteria for Phase 3

You have successfully completed Phase 3 when:

✅ Streamlit application is containerized and pushed to Artifact Registry
✅ GKE Autopilot cluster is created and running
✅ Application is deployed to Kubernetes with multiple replicas
✅ Load Balancer provides external access to application
✅ Horizontal Pod Autoscaler is configured and tested
✅ Monitoring and logging are set up in Cloud Console
✅ Alerts are configured for key metrics
✅ Security best practices are implemented
✅ Application scales automatically under load
✅ Complete documentation exists for operations and development
✅ You understand Kubernetes core concepts and GKE specifics

---

## Congratulations!

You have completed all three phases of the Titanic ML GCP project. You now have:

1. **Production-grade ML pipeline** with Feature Store, model training, and deployment
2. **MLOps capabilities** with experiment tracking, hyperparameter optimization, and model registry
3. **Cloud-native deployment** on Kubernetes with auto-scaling, monitoring, and security

## Next Steps for Further Learning

**Advanced GKE topics:**
- Service mesh (Istio/Anthos Service Mesh)
- Advanced networking (Network Policies, Ingress)
- Multi-cluster deployments
- GitOps with Argo CD or Flux

**Production hardening:**
- Implement comprehensive CI/CD
- Add integration and e2e tests
- Set up staging and production environments
- Implement blue-green or canary deployments

**ML enhancements:**
- Real-time retraining pipelines
- A/B testing framework
- Model monitoring and drift detection
- Feature engineering automation

**Cost and performance optimization:**
- More aggressive auto-scaling
- Caching strategies
- CDN for static assets
- Database for persistent data

**Compliance and governance:**
- Data encryption at rest and in transit
- Audit logging
- Access controls and RBAC
- Compliance certifications (SOC 2, HIPAA, etc.)

Great work completing this comprehensive ML engineering project! 🎉

