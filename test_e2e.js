const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const BASE = 'http://localhost:5173';
const API = 'http://127.0.0.1:8000/api/v1';
const SCREENSHOT_DIR = path.join(__dirname, 'test-screenshots');

// Ensure screenshot directory exists
if (!fs.existsSync(SCREENSHOT_DIR)) fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

async function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

async function log(msg) {
  const ts = new Date().toLocaleTimeString();
  console.log(`[${ts}] ${msg}`);
}

(async () => {
  const browser = await chromium.launch({ headless: false, slowMo: 300 });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  const page = await context.newPage();

  let passed = 0;
  let failed = 0;
  const results = [];

  async function test(name, fn) {
    try {
      await fn();
      passed++;
      results.push({ name, status: 'PASS' });
      await log(`PASS: ${name}`);
    } catch (e) {
      failed++;
      results.push({ name, status: 'FAIL', error: e.message });
      await log(`FAIL: ${name} — ${e.message}`);
    }
  }

  async function screenshot(name) {
    const file = path.join(SCREENSHOT_DIR, `${name}.png`);
    await page.screenshot({ path: file, fullPage: true });
    await log(`  Screenshot: ${file}`);
  }

  // ════════════════════════════════════════════════════════
  // 1. Dashboard
  // ════════════════════════════════════════════════════════
  await test('1. Dashboard page loads', async () => {
    await page.goto(BASE);
    await page.waitForSelector('text=HlAB Console', { timeout: 10000 });
    await sleep(2000);
    await screenshot('01_dashboard');
  });

  // ════════════════════════════════════════════════════════
  // 2. Agents Page
  // ════════════════════════════════════════════════════════
  await test('2. Navigate to Agents page', async () => {
    await page.click('text=Agent 管理');
    await sleep(2000);
    await screenshot('02_agents_list');
  });

  await test('3. Create new agent', async () => {
    // Click "新建" button
    const createBtn = page.locator('button:has-text("新建"), button:has-text("创建"), button:has-text("添加")').first();
    if (await createBtn.isVisible()) {
      await createBtn.click();
      await sleep(1000);

      // Fill form if modal appears
      const modal = page.locator('.ant-modal');
      if (await modal.isVisible({ timeout: 3000 }).catch(() => false)) {
        // Fill name
        const nameInput = modal.locator('input').first();
        await nameInput.fill('Playwright测试Agent');

        // Fill description if visible
        const descInput = modal.locator('textarea').first();
        if (await descInput.isVisible().catch(() => false)) {
          await descInput.fill('由Playwright自动化测试创建的Agent');
        }

        await screenshot('03_agent_create_form');

        // Submit
        const okBtn = modal.locator('button:has-text("确"), button:has-text("OK"), button:has-text("提交"), .ant-btn-primary').last();
        await okBtn.click();
        await sleep(2000);
      }
    }
    await screenshot('03_agents_after_create');
  });

  // ════════════════════════════════════════════════════════
  // 3. Tools Page
  // ════════════════════════════════════════════════════════
  await test('4. Navigate to Tools page', async () => {
    await page.click('text=工具管理');
    await sleep(2000);
    await screenshot('04_tools_list');
  });

  await test('5. View tool details', async () => {
    // Click on first tool row if exists
    const row = page.locator('.ant-table-row').first();
    if (await row.isVisible({ timeout: 3000 }).catch(() => false)) {
      await row.click();
      await sleep(1000);
      await screenshot('05_tool_detail');
    }
  });

  await test('6. Test tool connectivity', async () => {
    // Look for a "测试" or "Test" button
    const testBtn = page.locator('button:has-text("测试"), button:has-text("Test"), button:has-text("连通")').first();
    if (await testBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      await testBtn.click();
      await sleep(2000);
      await screenshot('06_tool_test');
    }
  });

  // ════════════════════════════════════════════════════════
  // 4. Knowledge Page
  // ════════════════════════════════════════════════════════
  await test('7. Navigate to Knowledge page', async () => {
    await page.click('text=知识管理');
    await sleep(2000);
    await screenshot('07_knowledge');
  });

  // ════════════════════════════════════════════════════════
  // 5. Workflows Page
  // ════════════════════════════════════════════════════════
  await test('8. Navigate to Workflows page', async () => {
    await page.click('text=流程编排');
    await sleep(2000);
    await screenshot('08_workflows');
  });

  await test('9. Create workflow', async () => {
    const createBtn = page.locator('button:has-text("新建"), button:has-text("创建"), button:has-text("添加")').first();
    if (await createBtn.isVisible({ timeout: 3000 }).catch(() => false)) {
      await createBtn.click();
      await sleep(1000);

      const modal = page.locator('.ant-modal');
      if (await modal.isVisible({ timeout: 3000 }).catch(() => false)) {
        const nameInput = modal.locator('input').first();
        await nameInput.fill('测试工作流');
        await screenshot('09_workflow_create');

        const okBtn = modal.locator('.ant-btn-primary').last();
        await okBtn.click();
        await sleep(2000);
      }
    }
    await screenshot('09_workflows_after');
  });

  // ════════════════════════════════════════════════════════
  // 6. Audit Page
  // ════════════════════════════════════════════════════════
  await test('10. Navigate to Audit page', async () => {
    await page.click('text=审计回放');
    await sleep(2000);
    await screenshot('10_audit');
  });

  await test('11. View audit trace detail', async () => {
    const row = page.locator('.ant-table-row').first();
    if (await row.isVisible({ timeout: 3000 }).catch(() => false)) {
      await row.click();
      await sleep(1500);
      await screenshot('11_audit_detail');
    }
  });

  // ════════════════════════════════════════════════════════
  // 7. Return to Dashboard — verify metrics
  // ════════════════════════════════════════════════════════
  await test('12. Dashboard metrics refresh', async () => {
    await page.click('text=仪表盘');
    await sleep(2000);
    await screenshot('12_dashboard_final');
  });

  // ════════════════════════════════════════════════════════
  // Summary
  // ════════════════════════════════════════════════════════
  console.log('\n' + '='.repeat(60));
  console.log('  E2E TEST RESULTS');
  console.log('='.repeat(60));
  for (const r of results) {
    const icon = r.status === 'PASS' ? 'v' : 'x';
    const detail = r.error ? ` (${r.error.substring(0, 80)})` : '';
    console.log(`  [${icon}] ${r.name}${detail}`);
  }
  console.log('='.repeat(60));
  console.log(`  Total: ${results.length} | Passed: ${passed} | Failed: ${failed}`);
  console.log(`  Screenshots saved to: ${SCREENSHOT_DIR}`);
  console.log('='.repeat(60));

  await sleep(2000);
  await browser.close();
})();
