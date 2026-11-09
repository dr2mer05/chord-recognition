const { chromium } = require('playwright');
const assert = require('node:assert/strict');
(async () => {
 const videoId = 'Test' + Date.now().toString().slice(-7);
 const browser = await chromium.launch({...(process.env.CHROMIUM_PATH ? {executablePath:process.env.CHROMIUM_PATH} : {}),headless:true,args:['--no-sandbox']});
 const width = Number(process.env.CHORDAPP_VIEWPORT_WIDTH || 834);
 const page = await browser.newPage({viewport:{width,height:width<600 ? 844 : 1112},hasTouch:true});
 const errors=[]; page.on('pageerror', e => errors.push(e.message));
 await page.route('https://www.youtube.com/iframe_api', route => route.fulfill({contentType:'text/javascript',body:`
 window.YT = {PlayerState:{PLAYING:1},Player: class {
  constructor(id, options){ this.raw=5; this.rate=1; this.options=options; window.__testPlayer=this; setTimeout(()=>options.events.onReady(),0); }
  getCurrentTime(){return this.raw;} getPlayerState(){return 2;} getPlaybackRate(){return this.rate;}
  seekTo(t){this.lastSeek=t;this.raw=t;} destroy(){}
 }}; window.onYouTubeIframeAPIReady();` }));
 await page.goto(process.env.CHORDAPP_TEST_URL || 'http://127.0.0.1:8765');
 await page.locator('#url').fill('https://youtu.be/' + videoId);
 await page.locator('#doYt').click();
 await page.waitForFunction(()=>document.querySelector('#sourceStatus').textContent.includes('연결된 음원이 없습니다'));
 assert(await page.locator('#sourceForm').evaluate(el=>el.open));
 await page.locator('#sourceFile').setInputFiles({name:'browser-synth.wav',mimeType:'audio/wav',buffer:synthWave()});
 await page.locator('#rightsBasis').fill('자동 생성한 합성 음원. 테스트에서 분석·보관 허용.');
 await page.locator('#rightsConfirmed').check(); await page.locator('#sameRecording').check();
 await page.locator('#videoOffset').fill('5');
 await page.locator('#registerSource').click();
 await page.waitForFunction(()=>document.querySelector('#ytStatus').textContent.includes('분석을 완료했습니다'),{},{timeout:120000});
 assert(await page.locator('#resultPanel').isVisible());
 const first = await page.evaluate(()=>({sourceKey:state.sourceKey, segmentCount:state.segments.length, offset:state.videoOffset}));
 assert(first.sourceKey.startsWith('linked:' + videoId + ':')); assert(first.segmentCount>0); assert.equal(first.offset,5);
 await page.locator('.chip[data-i="0"]').first().click();
 await page.locator('#editorChord').fill('Cmaj7');
 const save = await page.locator('#editorSave').boundingBox(); assert(save.height>=44 && save.width>=44);
 await page.locator('#editorSave').click();
 await page.waitForFunction(()=>state.segments[0].chord === 'Cmaj7');
 assert.equal(await page.evaluate(()=>state.segments[0].chord),'Cmaj7');
 await page.reload();
 await page.locator('#url').fill('https://www.youtube.com/watch?v=' + videoId + '&t=10');
 await page.locator('#doYt').click();
 await page.waitForFunction(()=>document.querySelector('#ytStatus').textContent.includes('저장된 분석을 열었습니다'));
 assert.equal(await page.evaluate(()=>state.segments[0].chord),'Cmaj7');
 const clock = await page.evaluate(()=> {
  stopSync();const p=state.ytPlayer; p.raw=10;p.rate=2;p.getPlayerState=()=>1;
  state.clock={raw:10,at:performance.now()-200,rate:2};
  const audioTime=playbackTime(); const tapTime=currentPlaybackTime();
  seek(Math.min(1,state.segments.length-1));return {audioTime,tapTime,seek:p.lastSeek,expected:state.segments[Math.min(1,state.segments.length-1)].start+5};
 });
 assert(Math.abs(clock.audioTime-5.4)<.1);assert.equal(clock.tapTime,5);assert.equal(clock.seek,clock.expected);
 const overflow = await page.evaluate(()=>document.documentElement.scrollWidth > innerWidth+1);
 assert.equal(overflow,false,'viewport horizontal overflow');
 await page.screenshot({path:process.env.CHORDAPP_SCREENSHOT || 'work/browser-preview.png',fullPage:true});
 await page.evaluate(()=>state.ytPlayer.options.events.onError({data:101}));
 assert((await page.locator('#playerStatus').textContent()).includes('외부 재생'));
 await page.locator('#sourceLibrary').evaluate(el=>el.closest('details').open=true);
 await page.locator('[data-revoke="0"]').click();
 await page.waitForFunction(()=>document.querySelector('#sourceStatus').textContent.includes('해제'));
 await page.locator('#doYt').click();
 await page.waitForFunction(()=>document.querySelector('#ytStatus').textContent.includes('해제'));
 assert.equal(errors.length,0,errors.join('\n'));
 console.log(JSON.stringify({passed:true,viewportWidth:width,checks:['missing source guidance','real upload and CPU analysis','correction persisted','reload uses cache','video offset and rate','seek and tap mapping','player error','revocation blocks cache','responsive layout'],...first,clock,pageErrors:errors}));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});

// 독립 생성한 8초 합성 진행. 실제 음악 정확도를 평가하는 데이터가 아니다.
function synthWave() {
 const sr=22050, count=8*sr, wav=Buffer.alloc(44+count*2);
 wav.write('RIFF');wav.writeUInt32LE(36+count*2,4);wav.write('WAVE',8);wav.write('fmt ',12);
 wav.writeUInt32LE(16,16);wav.writeUInt16LE(1,20);wav.writeUInt16LE(1,22);
 wav.writeUInt32LE(sr,24);wav.writeUInt32LE(sr*2,28);wav.writeUInt16LE(2,32);wav.writeUInt16LE(16,34);
 wav.write('data',36);wav.writeUInt32LE(count*2,40);
 const chords=[[60,64,67],[57,60,64],[53,57,60],[55,59,62]];
 for(let i=0;i<count;i++){const t=i/sr;const local=t%2;const env=Math.min(1,local/.02,(2-local)/.03);
  const value=chords[Math.floor(t/2)].reduce((v,m)=>v+Math.sin(2*Math.PI*440*Math.pow(2,(m-69)/12)*t),0)/3;
  wav.writeInt16LE(Math.round(value*env*12000),44+i*2);}
 return wav;
}
