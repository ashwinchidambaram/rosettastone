# Single Sign-On (SSO) Configuration Guide

## Overview

Meridian AI supports Single Sign-On (SSO) to streamline authentication for your organization. SSO allows your team to log in using your existing identity provider (IdP) credentials, eliminating the need for separate Meridian passwords.

## Supported Protocols

SSO is supported via SAML 2.0 and OpenID Connect (OIDC). Both protocols are fully supported, and you can choose the one that best matches your identity provider's capabilities.

## Tier Availability

SSO is included on Professional and Enterprise tiers. SSO is not available on the Starter tier. If you need SSO and are currently on the Starter plan, you will need to upgrade to Professional or higher.

## Prerequisites

To set up SSO, you will need the following information from your identity provider:

1. **IdP Metadata URL** -- The URL where Meridian can fetch your IdP's SAML metadata or OIDC discovery document.
2. **Entity ID** -- The unique identifier for your IdP application.
3. **X.509 Certificate** -- The certificate used by your IdP to sign SAML assertions (for SAML 2.0 setups).

## SAML 2.0 Setup Steps

1. Navigate to Settings > Security > SSO in the Meridian dashboard.
2. Select "SAML 2.0" as the protocol.
3. Enter your IdP Metadata URL.
4. Enter the Entity ID for your IdP application.
5. Upload or paste the X.509 certificate.
6. Click "Test Connection" to verify the setup.
7. Once the test passes, click "Enable SSO."

## OpenID Connect (OIDC) Setup Steps

1. Navigate to Settings > Security > SSO in the Meridian dashboard.
2. Select "OpenID Connect" as the protocol.
3. Enter the OIDC Discovery URL (e.g., `https://your-idp.com/.well-known/openid-configuration`).
4. Enter the Client ID and Client Secret from your IdP.
5. Click "Test Connection" to verify.
6. Enable SSO once the test passes.

## Just-In-Time (JIT) Provisioning

JIT (Just-In-Time) user provisioning is supported. When JIT is enabled, users are automatically created in Meridian on their first SSO login. There is no need to manually create user accounts ahead of time.

JIT-provisioned users inherit the default role configured in your SSO settings (typically "Viewer"). Admins can adjust roles after the user's first login.

## SCIM Provisioning

SCIM provisioning for automated user lifecycle management is available on the Enterprise tier only. SCIM allows your IdP to automatically create, update, and deactivate user accounts in Meridian based on changes in your directory.

To enable SCIM:
1. Navigate to Settings > Security > Provisioning.
2. Generate a SCIM API token.
3. Configure your IdP's SCIM integration with the Meridian SCIM endpoint and token.

## Troubleshooting SSO

- **Error: "Invalid SAML Response"** -- Verify that the X.509 certificate matches the one configured in your IdP.
- **Error: "Entity ID Mismatch"** -- Ensure the Entity ID in Meridian matches the one configured in your IdP application.
- **Users not being created on first login** -- Confirm that JIT provisioning is enabled in Settings > Security > SSO.
