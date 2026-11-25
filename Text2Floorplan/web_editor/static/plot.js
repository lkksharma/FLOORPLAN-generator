// Step 1: plot image -> detected outline -> a length in metres for every edge.
const $=id=>document.getElementById(id),svgNS='http://www.w3.org/2000/svg';
const LETTERS='ABCDEFGHIJKL',MAX_EDGES=12,MAX_SIDE=1400;
let image=null,points=[],values=[],dragging=null,active=null,checked=false;

const element=(name,attrs={},text)=>{const node=document.createElementNS(svgNS,name);for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);if(text!==undefined)node.textContent=text;return node;};
async function request(path,body){const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json','X-Text2Revit-Editor':'1'},body:JSON.stringify(body)});const result=await response.json();if(!response.ok)throw Error(result.error||'Request failed');return result;}

// --- image -----------------------------------------------------------------
async function load(file){
  $('image-error').textContent='';
  if(!file)return;
  if(!/^image\//.test(file.type)){$('image-error').textContent='Choose an image file (PNG, JPG or WebP).';return;}
  try{
    const bitmap=await createImageBitmap(file),scale=Math.min(1,MAX_SIDE/Math.max(bitmap.width,bitmap.height));
    const canvas=document.createElement('canvas');canvas.width=Math.round(bitmap.width*scale);canvas.height=Math.round(bitmap.height*scale);
    const context=canvas.getContext('2d');context.fillStyle='#fff';context.fillRect(0,0,canvas.width,canvas.height);context.drawImage(bitmap,0,0,canvas.width,canvas.height);
    let url=canvas.toDataURL('image/png');if(url.length>4e6)url=canvas.toDataURL('image/jpeg',.9);
    image={url,width:canvas.width,height:canvas.height};
  }catch{$('image-error').textContent='This image could not be opened.';return;}
  await detect();
}
async function detect(){
  $('drop').hidden=true;$('stage').hidden=false;$('hint').textContent='Finding the plot outline…';points=[];render();
  let message='';
  try{points=(await request('/api/plot/detect',{image:image.url})).points;}
  catch(error){
    message=error.message;const w=image.width,h=image.height;
    points=[[w*.2,h*.2],[w*.8,h*.2],[w*.8,h*.8],[w*.2,h*.8]];
  }
  values=points.map(()=>'');checked=false;active=null;
  $('image-error').textContent=message?message+' A starting rectangle was placed for you.':'';
  $('hint').textContent='Drag a corner to correct it. Click + to add a corner, double-click a corner to remove it.';
  $('lengths').hidden=false;document.querySelector('.work').classList.add('ready');
  render();fields();
}

// --- drawing ---------------------------------------------------------------
function problem(index){
  const text=values[index].trim().replace(',','.');
  if(!text)return 'Enter the length of this edge.';
  if(!/^\d*\.?\d+$/.test(text)||!(Number(text)>0))return 'Enter a number in metres, such as 14.5.';
  if(Number(text)<1||Number(text)>200)return 'Enter a length between 1 and 200 m.';
  return '';
}
function render(){
  const svg=$('canvas');svg.replaceChildren();if(!image)return;
  svg.setAttribute('viewBox',`0 0 ${image.width} ${image.height}`);
  svg.append(element('image',{href:image.url,width:image.width,height:image.height,opacity:.6}));
  if(!points.length)return;
  const rect=svg.getBoundingClientRect(),shown=rect.width?Math.min(rect.width/image.width,(rect.height||rect.width)/image.height):1,unit=1/(shown||1);
  svg.append(element('polygon',{class:'outline',points:points.map(p=>p.join(',')).join(' ')}));
  const cx=points.reduce((s,p)=>s+p[0],0)/points.length,cy=points.reduce((s,p)=>s+p[1],0)/points.length;
  const orientation=points.reduce((s,p,i)=>{const q=points[(i+1)%points.length];return s+p[0]*q[1]-q[0]*p[1]},0)>0?1:-1;
  points.forEach((a,i)=>{
    const b=points[(i+1)%points.length],length=Math.hypot(b[0]-a[0],b[1]-a[1])||1,n=[(b[1]-a[1])/length*orientation,-(b[0]-a[0])/length*orientation],mid=[(a[0]+b[0])/2,(a[1]+b[1])/2];
    const state=(active===i?' active':'')+(checked&&problem(i)?' bad':'');
    svg.append(element('line',{class:'edge'+state,x1:a[0],y1:a[1],x2:b[0],y2:b[1]}));
    // Letter outside the outline, kept inside the picture.
    const r=13*unit,x=Math.min(Math.max(mid[0]+n[0]*24*unit,r),image.width-r),y=Math.min(Math.max(mid[1]+n[1]*24*unit,r),image.height-r);
    const badge=element('g',{class:'badge'+state});badge.append(element('circle',{cx:x,cy:y,r}),element('text',{x,y,'font-size':14*unit},LETTERS[i]));
    badge.addEventListener('click',()=>document.querySelector(`#edge-${i}`)?.focus());svg.append(badge);
    if(points.length<MAX_EDGES&&length*shown>70){
      const add=element('g',{class:'add',role:'button',tabindex:0,'aria-label':`Add a corner on edge ${LETTERS[i]}`});
      add.append(element('circle',{cx:mid[0],cy:mid[1],r:8*unit}),element('text',{x:mid[0],y:mid[1],'font-size':13*unit},'+'));
      const insert=()=>{points.splice(i+1,0,mid);values.splice(i,1,'','');active=null;render();fields();};
      add.addEventListener('click',insert);add.addEventListener('keydown',event=>{if(event.key==='Enter')insert()});svg.append(add);
    }
  });
  points.forEach((p,i)=>{
    const corner=element('circle',{class:'corner',cx:p[0],cy:p[1],r:8*unit,tabindex:0,'aria-label':`Corner ${i+1}`});
    corner.addEventListener('pointerdown',event=>{event.preventDefault();dragging=i;$('canvas').setPointerCapture(event.pointerId)});
    corner.addEventListener('dblclick',()=>remove(i));
    corner.addEventListener('keydown',event=>{
      const step={ArrowLeft:[-2,0],ArrowRight:[2,0],ArrowUp:[0,-2],ArrowDown:[0,2]}[event.key];
      if(step){event.preventDefault();move(i,[p[0]+step[0],p[1]+step[1]]);$('canvas').querySelectorAll('.corner')[i]?.focus()}
      if(event.key==='Delete'||event.key==='Backspace'){event.preventDefault();remove(i)}
    });
    svg.append(corner);
  });
}
function move(index,point){points[index]=[Math.min(Math.max(point[0],0),image.width),Math.min(Math.max(point[1],0),image.height)];$('form-error').textContent='';render();}
function remove(index){
  if(points.length<=3){$('form-error').textContent='A plot needs at least three corners.';return;}
  const previous=(index+points.length-1)%points.length;
  points.splice(index,1);values[previous]='';values.splice(index,1);active=null;render();fields();
}
$('canvas').addEventListener('pointermove',event=>{if(dragging===null)return;const p=$('canvas').createSVGPoint();p.x=event.clientX;p.y=event.clientY;const q=p.matrixTransform($('canvas').getScreenCTM().inverse());move(dragging,[q.x,q.y])});
for(const name of ['pointerup','pointercancel'])$('canvas').addEventListener(name,()=>{dragging=null});
// Pointer capture retargets double clicks to the canvas, so find the corner by position.
$('canvas').addEventListener('dblclick',event=>{if(event.target!==$('canvas'))return;const p=$('canvas').createSVGPoint();p.x=event.clientX;p.y=event.clientY;const matrix=$('canvas').getScreenCTM(),q=p.matrixTransform(matrix.inverse()),index=points.findIndex(c=>Math.hypot(c[0]-q.x,c[1]-q.y)*matrix.a<14);if(index>=0)remove(index)});

// --- one text box per edge -------------------------------------------------
function fields(){
  $('fields').replaceChildren(...points.map((_,i)=>{
    const row=document.createElement('div');row.className='field';
    const letter=document.createElement('label');letter.className='letter';letter.htmlFor=`edge-${i}`;letter.textContent=LETTERS[i];
    const input=document.createElement('input');input.id=`edge-${i}`;input.inputMode='decimal';input.autocomplete='off';input.placeholder='Length';input.value=values[i];
    input.setAttribute('aria-label',`Length of edge ${LETTERS[i]} in metres`);input.setAttribute('aria-describedby',`problem-${i}`);
    const unit=document.createElement('span');unit.className='unit';unit.textContent='m';
    const note=document.createElement('p');note.className='problem';note.id=`problem-${i}`;
    input.oninput=()=>{values[i]=input.value;$('form-error').textContent='';show();};
    input.onfocus=()=>{active=i;render()};input.onblur=()=>{active=null;render()};
    row.append(letter,input,unit,note);return row;
  }));
  show();
}
function show(){
  [...$('fields').children].forEach((row,i)=>{const text=checked?problem(i):'';row.classList.toggle('bad',!!text);row.querySelector('.problem').textContent=text;row.querySelector('input').setAttribute('aria-invalid',String(!!text))});
  render();
}
$('lengths').onsubmit=async event=>{
  event.preventDefault();checked=true;$('form-error').textContent='';show();
  const first=values.findIndex((_,i)=>problem(i));
  if(first>=0){$(`edge-${first}`).focus();return;}
  const lengths=values.map(v=>Number(v.trim().replace(',','.'))),plot={points,lengths};
  $('next').disabled=true;
  try{
    const measured=await request('/api/plot/measure',plot);
    const saved={...plot,measured,image:{...image},values};
    try{sessionStorage.setItem('plot',JSON.stringify(saved));}
    catch{delete saved.image;sessionStorage.setItem('plot',JSON.stringify(saved));}
    location.href='/en';
  }catch(error){$('form-error').textContent=error.message;$('next').disabled=false;}
};

// --- start -----------------------------------------------------------------
$('file').onchange=()=>{load($('file').files[0]);$('file').value=''};
$('replace').onclick=()=>$('file').click();$('redetect').onclick=detect;
$('drop').onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();$('file').click()}};
for(const name of ['dragover','dragleave','drop'])document.addEventListener(name,event=>{event.preventDefault();$('drop').classList.toggle('over',name==='dragover');if(name==='drop')load(event.dataTransfer.files[0])});
addEventListener('resize',render);
if(new URLSearchParams(location.search).has('edit')){
  try{const saved=JSON.parse(sessionStorage.getItem('plot'));if(saved?.image){image=saved.image;points=saved.points;values=saved.values;$('drop').hidden=true;$('stage').hidden=false;$('lengths').hidden=false;document.querySelector('.work').classList.add('ready');fields();}}catch{}
}
