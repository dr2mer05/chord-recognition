// 네트워크 결과만 모사한다. 실제 비교 API·WAV 검증·화면·새로고침을 확인한다.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 try {
  const page=await browser.newPage({viewport:{width:390,height:844},hasTouch:true});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(process.env.CHORDAPP_TEST_URL||'http://127.0.0.1:8778');
  await page.locator('#openSettings').click();
  await page.locator('#compareConnection').click();
  await page.waitForFunction(()=>document.querySelector('#comparisonResult').textContent.includes('먼저 위의'));
  await page.locator('#url').fill('https://youtu.be/PrivateTest');
  await page.locator('#compareConnection').click();
  await page.waitForFunction(()=>document.querySelector('#comparisonResult').textContent.includes('비교 완료'));
  let text=await page.locator('#comparisonResult').textContent();
  assert(text.includes('기존 클라이언트 설정: 성공') && text.includes('현재 설정: 실패'));
  assert(text.includes('HTTP: 403') && text.includes('음원 다운로드'));
  await page.reload();await page.locator('#openSettings').click();
  await page.waitForFunction(()=>document.querySelector('#comparisonResult').textContent.includes('비교 완료'));
  for(const width of [390,834]) {
   await page.setViewportSize({width,height:width===390?844:1112});
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   assert((await page.locator('#compareConnection').boundingBox()).height>=44);
   await page.screenshot({path:`/workspace/work/connection-comparison-${width}.png`,fullPage:true});
  }
  assert.deepEqual(errors,[]);
  console.log('390/834px: two-profile comparison, separate outcomes, HTTP stage evidence and reload passed (simulated network)');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
