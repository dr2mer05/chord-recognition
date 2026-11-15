// Google 요청·YouTube 플레이어는 모사하고 설정 저장·검색·실제 DSP는 서버에서 실행한다.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH||'/usr/bin/chromium',headless:true,args:['--no-sandbox']});
 try {
  const page=await browser.newPage({viewport:{width:390,height:844},hasTouch:true});
  const errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('https://www.youtube.com/iframe_api',route=>route.fulfill({contentType:'text/javascript',body:`
   window.YT={PlayerState:{PLAYING:1},Player:class{constructor(id,o){setTimeout(()=>o.events.onReady(),0);}getCurrentTime(){return 0;}getPlayerState(){return 2;}getPlaybackRate(){return 1;}seekTo(){}destroy(){}}};window.onYouTubeIframeAPIReady();`}));
  await page.goto(process.env.CHORDAPP_TEST_URL||'http://127.0.0.1:8768');
  await page.waitForFunction(()=>document.querySelector('#q').disabled);
  await page.locator('#openSettings').click();
  await page.locator('#youtubeApiKey').fill('AIza'+'y'.repeat(35));
  await page.locator('#saveApiKey').click();
  await page.waitForFunction(()=>document.querySelector('#keyStatus').textContent.includes('API 키를 확인'));
  assert.equal(await page.locator('#youtubeApiKey').inputValue(),'');
  const key='AIza'+'x'.repeat(35);
  await page.locator('#youtubeApiKey').fill(key);await page.locator('#saveApiKey').click();
  await page.waitForFunction(()=>!document.querySelector('#q').disabled);
  assert.equal(await page.locator('#youtubeApiKey').inputValue(),'');
  assert(!(await page.locator('body').textContent()).includes(key));
  await page.locator('#q').fill('검증 곡');await page.locator('#doSearch').click();
  await page.waitForFunction(()=>document.querySelector('#searchStatus').textContent==='1건');
  assert.equal(await page.locator('#results .t').textContent(),'검색 검증 곡 <테스트>');
  await page.locator('#results button').click();
  await page.waitForFunction(()=>document.querySelector('#ytStatus').textContent.includes('분석을 완료했습니다'),{},{timeout:120000});
  assert(await page.locator('#resultPanel').isVisible());
  await page.reload();
  await page.waitForFunction(()=>!document.querySelector('#q').disabled);
  await page.locator('#openSettings').click();
  await page.waitForFunction(()=>document.querySelector('#diagnostics').textContent.includes('앱 버전'));
  assert(!(await page.locator('#diagnostics').textContent()).includes(key));
  assert.equal(await page.locator('#youtubeApiKey').inputValue(),'');
  for(const width of [390,834]) {
   await page.setViewportSize({width,height:width===390?844:1112});
   assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
   const box=await page.locator('#saveApiKey').boundingBox();assert(box.height>=44);
   await page.screenshot({path:`/workspace/work/nov11-search-${width}.png`,fullPage:true});
  }
  assert.deepEqual(errors,[]);
  console.log('390/834px: invalid key, secret clearing, key save, immediate search, escaped result, DSP analysis, reload persistence and diagnostics passed');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
