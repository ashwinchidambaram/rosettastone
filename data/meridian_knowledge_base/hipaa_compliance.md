# HIPAA Compliance & BAA Details

## Overview

Meridian AI supports healthcare organizations that need to comply with HIPAA regulations. Our platform provides the security controls, audit capabilities, and operational safeguards required for handling Protected Health Information (PHI).

## Business Associate Agreement (BAA)

A BAA is available for Enterprise tier customers only. The BAA is signed during the onboarding process as part of the Enterprise contract. The BAA covers all data processing activities within the Meridian platform.

If you are a healthcare organization or handle PHI, you must be on the Enterprise tier and have a signed BAA before storing any PHI in Meridian.

## HIPAA-Compliant Workspace

PHI must be stored in a dedicated HIPAA-compliant workspace that is separate from standard workspaces. The HIPAA workspace has additional security controls enabled by default:

- **Data export restrictions:** Export via CSV and Excel is disabled by default in HIPAA workspaces. This restriction can be overridden by a compliance admin if your organization's policies allow data export.
- **Session timeout:** Session timeout is set to 15 minutes for HIPAA workspaces (compared to 30 minutes for standard workspaces).
- **Enhanced audit logging:** All data access, modifications, and exports are logged with additional detail.

## Audit Logs

Audit logs for HIPAA workspaces are retained for 7 years and are immutable. This means audit logs cannot be modified or deleted, even by workspace administrators. The 7-year retention period meets HIPAA's minimum retention requirements.

Audit logs can be exported for external compliance review via Settings > Compliance > Export Audit Logs.

## Encryption

HIPAA workspaces enforce the following encryption standards:

- **Minimum TLS 1.2** is enforced for all connections. TLS 1.3 is preferred and used by default.
- Data at rest is encrypted with AES-256. Customer-managed keys (CMK) are recommended for HIPAA workspaces.

## Access Controls

HIPAA workspaces support all access control features from the Data Governance module:
- Column-level access controls
- Row-level security policies
- IP allowlisting
- Multi-factor authentication (MFA) enforcement

## Compliance Monitoring

Meridian provides a HIPAA Compliance Dashboard in the workspace that shows:
- Current compliance status across all controls
- Any configuration drift or policy violations
- Upcoming audit milestones and deadlines
