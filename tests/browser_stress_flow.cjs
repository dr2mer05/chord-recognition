// 실제 합성 WAV·CPU 분석·브라우저 재생을 사용한다. 실제 Safari 시험은 아니다.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({executablePath:process.env.CHROMIUM_PATH || undefined,
    headless:true,args:['--no-sandbox']});
  try {
    for (const width of [390,834]) {
      const page = await browser.newPage({viewport:{width,height:1000},hasTouch:true});
      const errors=[];page.on('pageerror',e=>errors.push(e.message));
      await page.goto(process.env.CHORDAPP_TEST_URL || 'http://127.0.0.1:8765');
      await page.locator('#stressPanel summary').click();
      await page.waitForFunction(()=>document.querySelectorAll('#stressDemos button').length===7);
      const button=page.locator('#stressDemos [data-n="one_beat_changes"]');
      const size=await button.boundingBox();assert(size.height>=44 && size.width>=44);
      await button.click();
      await page.waitForFunction(()=>document.querySelector('#stressDemoStatus').textContent.includes('화음 종류'),{},{timeout:120000});
      assert(await page.locator('#resultPanel').isVisible());
      assert((await page.locator('#stressReference').textContent()).includes('C → G → Am → F'));
      assert((await page.locator('#stressDemoStatus').textContent()).includes('경계 F'));
      assert(await page.locator('.chip').count()>8);
      await page.locator('#player').evaluate(async el=>{await el.play();});
      await page.waitForFunction(()=>document.querySelector('#player').currentTime>.2);
      assert.equal(await page.locator('#player').evaluate(el=>el.paused),false);
      await page.locator('#player').evaluate(el=>el.pause());
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      await page.screenshot({path:`work/stress-${width}.png`,fullPage:true});
      assert.deepEqual(errors,[]);
      console.log(JSON.stringify({passed:true,width,checks:['7 stress tracks','touch target','real CPU chart',
        'independent reference','quality and boundary metrics','audio playback','no horizontal overflow','no JS errors']}));
      await page.close();
    }
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
