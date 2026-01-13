const puppeteer = require('puppeteer');
const path = require('path');
const os = require('os');

async function captureOkky() {
    const desktopPath = path.join(os.homedir(), 'Desktop');
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const screenshotPath = path.join(desktopPath, `okky_screenshot_${timestamp}.png`);
    
    const browser = await puppeteer.launch({ headless: true });
    const page = await browser.newPage();
    await page.setViewport({ width: 1920, height: 1080 });
    await page.goto('http://okky.kr', { waitUntil: 'networkidle2' });
    await page.screenshot({ path: screenshotPath });
    await browser.close();
    
    console.log(`Screenshot saved: ${screenshotPath}`);
}

captureOkky().catch(console.error);
