# Data Encryption -- At Rest & In Transit

## Overview

Meridian AI encrypts all customer data both at rest and in transit, ensuring your data is protected at every stage.

## Data at Rest

All data at rest is encrypted using AES-256 encryption, one of the strongest encryption standards available. Encryption keys are managed by AWS Key Management Service (KMS) by default.

### Customer-Managed Keys (CMK)

Customer-managed keys (CMK) are available on the Enterprise tier. With CMK, you bring your own encryption keys from AWS KMS. This gives you full control over key lifecycle, rotation, and revocation. If you revoke a CMK, Meridian will be unable to access your encrypted data.

## Data in Transit

All data in transit is encrypted using TLS (Transport Layer Security):

- **TLS 1.3:** Enforced as the primary protocol for all connections.
- **TLS 1.2:** Accepted for backward compatibility with older clients.
- **TLS 1.1 and below:** Rejected. Connections using TLS 1.1 or earlier will be refused.

All API endpoints, dashboard access, and data sync connections enforce these TLS requirements.

## Backup Encryption

Database backups are encrypted with a separate encryption key. Backup key rotation occurs on a 90-day cycle. Backups are stored in a separate AWS region from the primary data for disaster recovery purposes.

## Field-Level Encryption

Field-level encryption is available via the Data Governance module for sensitive columns. With field-level encryption, specific columns (such as SSN, credit card numbers, or other PII) are encrypted at the field level, providing an additional layer of protection beyond full-disk encryption.

Field-level encrypted data can only be decrypted by users with explicit access permissions configured in the Data Governance module.
