#!/bin/bash
# Rendered into stack.json UserData. CloudFormation replaces only the @@...@@ tokens.
set -euo pipefail
umask 077
dnf install -y aws-cfn-bootstrap
trap '/opt/aws/bin/cfn-signal -e $? --stack "@@STACK@@" --resource AppInstance --region "@@REGION@@"' EXIT
dnf install -y python3.12 python3.12-pip xfsprogs curl-minimal tar gzip util-linux
systemctl enable --now amazon-ssm-agent
id askiisc >/dev/null 2>&1 || useradd --system --home-dir /var/lib/askiisc --shell /sbin/nologin askiisc
id caddy >/dev/null 2>&1 || useradd --system --home-dir /var/lib/askiisc/caddy --shell /sbin/nologin caddy
DATA_VOLUME='@@VOLUME@@'
SERIAL=$(printf '%s' "$DATA_VOLUME" | tr -d '-')
DEVICE=''
for attempt in $(seq 1 120); do
  DEVICE=$(lsblk -ndo PATH,SERIAL | awk -v serial="$SERIAL" '$2==serial {print $1}')
  if [ -n "$DEVICE" ]; then break; fi
  sleep 2
done
[ -n "$DEVICE" ] || { echo 'Persistent EBS device did not attach'; exit 1; }
if ! blkid "$DEVICE"; then mkfs.xfs "$DEVICE"; fi
UUID=$(blkid -s UUID -o value "$DEVICE")
mkdir -p /var/lib/askiisc
if ! grep -q "UUID=$UUID " /etc/fstab; then printf 'UUID=%s /var/lib/askiisc xfs defaults 0 2\n' "$UUID" >> /etc/fstab; fi
mountpoint -q /var/lib/askiisc || mount /var/lib/askiisc
chown root:askiisc /var/lib/askiisc
chmod 0711 /var/lib/askiisc
install -d -m 0750 -o root -g askiisc /var/lib/askiisc/releases /var/lib/askiisc/venvs /etc/askiisc /etc/askiisc/releases
install -d -m 0700 -o askiisc -g askiisc /var/lib/askiisc/auth /var/lib/askiisc/indexes
install -d -m 0700 /var/lib/askiisc/backups /usr/local/lib/askiisc
install -d -m 0700 -o caddy -g caddy /var/lib/askiisc/caddy
# Swap covers installation/startup spikes on the small instance. T3 CPU credits remain in standard mode.
if [ ! -f /var/lib/askiisc/swapfile ]; then
  dd if=/dev/zero of=/var/lib/askiisc/swapfile bs=1M count=2048 status=none
  chmod 0600 /var/lib/askiisc/swapfile
  mkswap /var/lib/askiisc/swapfile
fi
if ! grep -q '/var/lib/askiisc/swapfile ' /etc/fstab; then echo '/var/lib/askiisc/swapfile none swap sw 0 0' >> /etc/fstab; fi
swapon -a
SITE_HOST='@@DOMAIN@@'
TLS_CONFIG=''
if [ -z "$SITE_HOST" ]; then
  SITE_HOST='@@PUBLIC_IP@@'
  # IP addresses otherwise get an untrusted internal certificate by default.
  # Explicit ACME issuer: never fall back to a self-signed certificate.
  TLS_CONFIG='  tls {
    issuer acme https://acme-v02.api.letsencrypt.org/directory {
      profile shortlived
    }
  }'
fi
cat > /etc/askiisc/host.json <<EOF
{"domain":"$SITE_HOST","region":"@@REGION@@","bucket":"@@BUCKET@@","parameter":"@@PARAMETER@@"}
EOF
chown root:askiisc /etc/askiisc/host.json
chmod 0640 /etc/askiisc/host.json
# Official release archive, pinned by SHA-512. No unversioned install script.
curl --fail --location --retry 4 https://github.com/caddyserver/caddy/releases/download/v2.11.7/caddy_2.11.7_linux_amd64.tar.gz -o /tmp/caddy.tar.gz
printf '%s  /tmp/caddy.tar.gz\n' 'a7a433a1b133efc3c8d10eb0b99d52a24b5ef5c322dc77f5282182b1c0402139ab83f3a99f0c52409df77d20123fb0b523edad8a66d8f5e49136197bf61ef0e7' | sha512sum -c -
tar -xzf /tmp/caddy.tar.gz -C /usr/local/bin caddy
chmod 0755 /usr/local/bin/caddy
rm /tmp/caddy.tar.gz
install -d -m 0755 /etc/caddy
cat > /etc/caddy/Caddyfile <<EOF
{
  default_sni $SITE_HOST
}
$SITE_HOST {
$TLS_CONFIG
  encode zstd gzip
  @internal path /metrics /docs /redoc /openapi.json
  respond @internal 404
  header {
    X-Content-Type-Options nosniff
    Referrer-Policy same-origin
    Strict-Transport-Security "max-age=31536000"
  }
  request_body {
    max_size 32KB
  }
  reverse_proxy 127.0.0.1:8100 {
    flush_interval -1
  }
}
EOF
chmod 0644 /etc/caddy/Caddyfile
cat > /usr/local/bin/askiisc-start <<'EOF'
#!/bin/bash
set -euo pipefail
mountpoint -q /var/lib/askiisc
release=$(readlink -f /var/lib/askiisc/current)
lockhash=$(sha256sum "$release/deploy/aws/requirements.lock" | cut -d ' ' -f 1)
exec "/var/lib/askiisc/venvs/$lockhash/bin/python" "$release/deploy/aws/runtime.py" --release "$release"
EOF
chmod 0755 /usr/local/bin/askiisc-start
cat > /etc/systemd/system/askiisc.service <<'EOF'
[Unit]
Description=Ask IISc application
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/var/lib/askiisc
ConditionPathExists=/var/lib/askiisc/current/release.json
[Service]
Type=simple
User=askiisc
Group=askiisc
ExecStart=/usr/local/bin/askiisc-start
Environment=PYTHONDONTWRITEBYTECODE=1
Restart=on-failure
RestartSec=5
TimeoutStopSec=45
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/askiisc/auth /var/lib/askiisc/indexes
[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/caddy.service <<'EOF'
[Unit]
Description=Ask IISc HTTPS proxy
After=network-online.target
Wants=network-online.target
RequiresMountsFor=/var/lib/askiisc
[Service]
Type=notify
User=caddy
Group=caddy
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
ExecReload=/usr/local/bin/caddy reload --config /etc/caddy/Caddyfile --adapter caddyfile
Environment=HOME=/var/lib/askiisc/caddy
Environment=XDG_DATA_HOME=/var/lib/askiisc/caddy/data
Environment=XDG_CONFIG_HOME=/var/lib/askiisc/caddy/config
Restart=on-failure
AmbientCapabilities=CAP_NET_BIND_SERVICE
CapabilityBoundingSet=CAP_NET_BIND_SERVICE
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/var/lib/askiisc/caddy
[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/askiisc-backup.service <<'EOF'
[Unit]
Description=Back up Ask IISc OTP sessions to private S3
RequiresMountsFor=/var/lib/askiisc
ConditionPathExists=/usr/local/lib/askiisc/host.py
[Service]
Type=oneshot
ExecStart=/usr/bin/python3.12 /usr/local/lib/askiisc/host.py backup
UMask=0077
EOF
cat > /etc/systemd/system/askiisc-backup.timer <<'EOF'
[Unit]
Description=Daily Ask IISc state backup
[Timer]
OnCalendar=*-*-* 03:00:00 UTC
Persistent=true
RandomizedDelaySec=300
[Install]
WantedBy=timers.target
EOF
mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nSystemMaxUse=128M\nMaxRetentionSec=7day\n' > /etc/systemd/journald.conf.d/askiisc.conf
systemctl restart systemd-journald
systemctl daemon-reload
systemctl enable askiisc.service caddy.service
# The release command starts services after secrets and a release are installed.
echo 'Host provisioned. Store the runtime secret, deploy a release, then verify trusted HTTPS at the stack ApplicationUrl.'
