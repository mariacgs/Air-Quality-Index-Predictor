# 🌍 Air Quality Index Predictor - Infrastructure Setup Log
**Role:** AWS Infrastructure & Deployment
**Platform:** AWS Academy (Learner Labs)
**Tools Used:** AWS CLI v2, AWS SAM CLI, Python 3.12, YAML

---

## 📋 Overview of What We Accomplished
We designed, configured, debugged, and prepared a multi-model serverless deployment blueprint (`template.yaml`) tailored specifically for the security constraints of the **AWS Academy Sandbox** environment. The architecture links a public-facing API Gateway endpoint to two distinct AWS Lambda functions running optimized ML models (XGBoost and LightGBM).

---

## 🛠️ Step-by-Step Implementation Ledger

### Step 1: Local Environment Configuration
1. **Verified AWS CLI:** Confirmed local installation of the AWS Command Line Interface (`aws-cli/2.34.48`).
2. **Installed AWS SAM CLI:** Downloaded, unzipped, and installed the official Linux AWS Serverless Application Model (SAM) CLI tool for automated infrastructure deployments.
3. **AWS Academy Session Token Linkage:** Configured temporary CLI environment variables (`aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`) extracted from the Learner Lab console to authenticate the local terminal.

### Step 2: Infrastructure-as-Code (IaC) Blueprint Engineering

Authored a customized template.yaml configuration file utilizing SAM shortcuts. Key choices made:

    Dynamic Parameterization: Implemented a MemSizeForLambda dropdown parameter (256MB, 512MB, 1024MB) allowing Person 3 to dynamically cycle memory allocations during load testing without editing code.

    Serverless Triggers: Configured integrated HTTP POST events mapping paths directly to routing layers:

        /predict/xgb ➡️ XGBoostPredictorFunction

        /predict/lgbm ➡️ LightGBMPredictorFunction

### Step 4: Troubleshooting & Sandbox Alignment

During initial iterations, we encountered and resolved critical architectural block:

    The AWS Academy IAM Lockdown (403 Unauthorized):

        Symptom: CloudFormation builds failed with is not authorized to perform: iam:CreateRole.

        Resolution: Circumvented sandbox security boundaries by hardcoding the pre-provisioned institutional execution role directly into the properties of both Lambda functions using:
        Role: !Sub "arn:aws:iam::${AWS::AccountId}:role/LabRole"

# AWS SAM Deployment Steps & Workarounds

This document outlines the step-by-step process used to successfully build and deploy the **Air Quality Index Predictor** serverless application using AWS SAM in a restricted AWS Academy / Vocareum lab environment.

## Project Architecture Summary
* **Runtime:** Python 3.12
* **Infrastructure:** AWS API Gateway + 2 AWS Lambda Functions (`XGBoostPredictorFunction`, `LightGBMPredictorFunction`)
* **Dependencies:** `numpy`, `onnxruntime`

---

## Step 1: Building the Code Locally
The deployment process starts by packaging the source code and installing the required Python dependencies specified in `requirements.txt`.

```bash
sam build 
```

### Step 3: Explicit Manual Deployment

With the correct bucket name, region definitions, and updated session credentials in place, the application was successfully deployed bypassing the automated wizard:

```bash
sam deploy \
  --stack-name aqi-predictor-test \
  --s3-bucket aws-sam-cli-managed-default-samclisourcebucket-wspaivggerrl \
  --capabilities CAPABILITY_IAM \
  --region us-east-1
```

    Troubleshooting Note: During this phase, if a previous deployment fails during creation, CloudFormation freezes the stack state in ROLLBACK_COMPLETE. Before rerunning the deploy command above, the broken footprint must be cleared with:
    Bash

```bash
aws cloudformation delete-stack --stack-name aqi-predictor-test --region us-east-1
```

### Step 4: Extracting the Live API Gateway Endpoint

Because the deployment bypassed the automated wizard, the output parameters did not print automatically. The unique API ID was safely pulled directly via the AWS CLI:

```bash
aws apigateway get-rest-apis --region us-east-1 --query "items[?name=='aqi-predictor-test'].id
```