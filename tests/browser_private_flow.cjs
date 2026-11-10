// 실제 분석 엔진 + 합성 공급 음원 + 가짜 YouTube 플레이어를 사용하는 모바일 회귀.
const { chromium } = require('playwright');
const assert = require('node:assert/strict');
(async () => {
 const browser = await chromium.launch({...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH}:{}), headless:true,args:['--no-sandbox']});
 try {
  const page = await browser.newPage({viewport:{width:390,height:844},hasTouch:true});
  const errors=[]; page.on('pageerror', e=>errors.push(e.message));
  await page.route('https://www.youtube.com/iframe_api', route=>route.fulfill({contentType:'text/javascript',body:`
   window.YT={PlayerState:{PLAYING:1},Player:class {
    constructor(id,options){this.options=options;this.raw=0;setTimeout(()=>options.events.onReady(),0);}
    getCurrentTime(){return this.raw;}getPlayerState(){return 2;}getPlaybackRate(){return 1;}
    seekTo(t){this.raw=t;}destroy(){}
   }};window.onYouTubeIframeAPIReady();`}));
  await page.goto(process.env.CHORDAPP_TEST_URL || 'http://127.0.0.1:8766');
  const video='https://youtu.be/PrivateTest';
  await page.locator('#url').fill(video);await page.locator('#doYt').click();
  await page.waitForFunction(()=>document.querySelector('#ytStatus').textContent.includes('분석을 완료했습니다'),{},{timeout:120000});
  assert(await page.locator('#resultPanel').isVisible());
  assert.equal(await page.locator('#sourceForm').evaluate(el=>el.open),false);
  assert((await page.locator('#analysisOrigin').textContent()).includes('음원은 삭제'));
  assert((await page.evaluate(()=>state.sourceKey)).startsWith('private-youtube:'));
  await page.locator('.chip[data-i="0"]').first().click();
  await page.locator('#editorChord').fill('Am7');await page.locator('#editorSave').click();
  await page.waitForFunction(()=>state.segments[0].chord==='Am7');
  await page.reload();await page.locator('#url').fill(video);await page.locator('#doYt').click();
  await page.waitForFunction(()=>document.querySelector('#ytStatus').textContent.includes('저장된 분석을 열었습니다'));
  assert.equal(await page.evaluate(()=>state.segments[0].chord),'Am7');
  await page.waitForFunction(()=>state.ytPlayer);
  const expected=await page.evaluate(()=>state.segments[Math.min(1,state.segments.length-1)].start);
  const actual=await page.evaluate(()=>{seek(Math.min(1,state.segments.length-1));return state.ytPlayer.getCurrentTime();});
  assert.equal(actual,expected);
  assert(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth));
  await page.screenshot({path:process.env.CHORDAPP_SCREENSHOT || '/tmp/chord-private-mobile.png',fullPage:true});
  assert.deepEqual(errors,[]);console.log('390px private link: no upload, DSP chart, correction reload, cache and seek passed');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exit(1);});
