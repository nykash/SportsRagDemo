#!/usr/bin/env node

/**
 * Script to generate config.js from .env file
 * Usage: node generate_config.js [path/to/.env]
 * 
 * This reads the .env file and generates a config.js file with AWS credentials
 */

const fs = require('fs');
const path = require('path');

// Get .env file path (default to ../backend/.env)
const envPath = process.argv[2] || path.join(__dirname, '../backend/.env');

if (!fs.existsSync(envPath)) {
    console.error(`Error: .env file not found at ${envPath}`);
    console.error('Usage: node generate_config.js [path/to/.env]');
    process.exit(1);
}

// Read .env file
const envContent = fs.readFileSync(envPath, 'utf8');
const envVars = {};

// Parse .env file
envContent.split('\n').forEach(line => {
    line = line.trim();
    if (line && !line.startsWith('#')) {
        const [key, ...valueParts] = line.split('=');
        if (key && valueParts.length > 0) {
            const value = valueParts.join('=').trim();
            // Remove quotes if present
            envVars[key.trim()] = value.replace(/^["']|["']$/g, '');
        }
    }
});

// Generate config.js
const configJs = `// S3 Configuration
// Auto-generated from .env file
// DO NOT commit this file with real credentials to version control

window.AWS_CONFIG = {
    bucketName: '${envVars.AWS_BUCKET_NAME || ''}',
    region: '${envVars.AWS_REGION || 'us-east-1'}',
    accessKeyId: '${envVars.AWS_ACCESS_KEY_ID || ''}',
    secretAccessKey: '${envVars.AWS_SECRET_ACCESS_KEY || ''}'
};
`;

// Write config.js
const configPath = path.join(__dirname, 'config.js');
fs.writeFileSync(configPath, configJs);
console.log(`✓ Generated config.js from ${envPath}`);
console.log(`  Make sure config.js is in .gitignore!`);

