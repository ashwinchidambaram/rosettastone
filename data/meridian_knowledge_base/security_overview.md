# Security Architecture & Certifications

## Overview

Meridian AI takes security seriously. Our platform is built with a defense-in-depth approach, combining multiple layers of security controls to protect customer data.

## Certifications

### SOC 2 Type II
Meridian AI has been SOC 2 Type II certified since 2023. The SOC 2 Type II audit report is available to customers under NDA. Contact your account manager or email security@meridian-ai.com to request a copy.

### HIPAA
Meridian AI supports HIPAA compliance through Business Associate Agreements (BAA) for Enterprise tier customers. See the HIPAA Compliance documentation for details.

## Penetration Testing

Penetration testing is conducted annually by a third-party security firm. The penetration test report is available to Enterprise customers upon request. We also conduct internal security assessments quarterly.

## Infrastructure

### Hosting
All data is hosted on AWS in the us-east-1 (N. Virginia) and eu-west-1 (Ireland) regions. Enterprise customers can select their preferred region during workspace setup.

### Tenancy Model
- **Enterprise tier:** Single-tenant infrastructure. Your data is hosted on dedicated compute and storage resources isolated from other customers.
- **Starter and Professional tiers:** Multi-tenant infrastructure with logical isolation between workspaces.

## Network Security

- All traffic is encrypted in transit using TLS 1.3 (TLS 1.2 also accepted).
- WAF (Web Application Firewall) protects against common web exploits.
- DDoS protection is provided via AWS Shield.
- IP allowlisting is available for Enterprise customers to restrict access to specific IP ranges.

## Bug Bounty Program

Meridian AI maintains a responsible disclosure program. Security researchers can report vulnerabilities via email to security@meridian-ai.com. We respond to all reports within 48 hours and offer bounties for qualifying vulnerabilities.

## Compliance and Privacy

- Data processing complies with GDPR, CCPA, and other applicable regulations.
- Customer data is never used for training ML models or shared with third parties.
- Data deletion requests are processed within 30 days per our data retention policy.
