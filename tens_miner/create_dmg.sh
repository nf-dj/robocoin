#!/bin/bash

# Set variables
APP_NAME="TensMiner"
DMG_NAME="$APP_NAME.dmg"
TMP_DMG="tmp.dmg"
VOLUME_NAME="$APP_NAME Installer"

# Create a temporary DMG
hdiutil create -size 200m -fs HFS+ -volname "$VOLUME_NAME" "$TMP_DMG"

# Mount the DMG
hdiutil attach "$TMP_DMG"

# Copy the app
cp -r "dist/$APP_NAME.app" "/Volumes/$VOLUME_NAME/"

# Create Applications symlink
ln -s /Applications "/Volumes/$VOLUME_NAME/Applications"

# Unmount
hdiutil detach "/Volumes/$VOLUME_NAME"

# Convert to compressed DMG
hdiutil convert "$TMP_DMG" -format UDZO -o "dist/$DMG_NAME"

# Clean up
rm "$TMP_DMG"

echo "Created dist/$DMG_NAME"