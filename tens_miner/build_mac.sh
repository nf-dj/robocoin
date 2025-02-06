#!/bin/bash

# Exit on error
set -e

echo "Installing dependencies..."
pip install -r requirements.txt
pip install pyinstaller dmgbuild biplist

echo "Building application..."
pyinstaller tens_miner.spec

echo "Creating app bundle..."
mkdir -p "dist/TensMiner.app/Contents/MacOS"
mkdir -p "dist/TensMiner.app/Contents/Resources"

# Copy executable
mv "dist/TensMiner" "dist/TensMiner.app/Contents/MacOS/"

# Create Info.plist
cat > "dist/TensMiner.app/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDisplayName</key>
    <string>TensMiner</string>
    <key>CFBundleExecutable</key>
    <string>TensMiner</string>
    <key>CFBundleIconFile</key>
    <string>icon.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.tenscoin.miner</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>TensMiner</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.13</string>
    <key>NSHighResolutionCapable</key>
    <true/>
</dict>
</plist>
EOF

# Optional: Copy icon if it exists
if [ -f "tens_miner/icon.icns" ]; then
    cp "tens_miner/icon.icns" "dist/TensMiner.app/Contents/Resources/"
fi

# Make executable
chmod +x "dist/TensMiner.app/Contents/MacOS/TensMiner"

echo "Creating DMG..."
dmgbuild -s dmg_settings.py "TensMiner" "dist/TensMiner.dmg"

echo "Build complete!"
echo "Created: dist/TensMiner.app"
echo "Created: dist/TensMiner.dmg"