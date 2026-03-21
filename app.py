import os, json, base64
from io import BytesIO
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import gdown

st.set_page_config(page_title='YOLO Detection', layout='wide')

FILE_ID    = '1Ehy-UdgyYk-O4XLZouEyGKj3opOtiMac'  # <-- замените на свой
MODEL_PATH = 'best.pt'

# Загрузка модели
@st.cache_resource
def load_model():
    # Скачиваем модель с Google Drive только при первом запуске
    if not os.path.exists(MODEL_PATH):
        gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

# Canvas с hover-тултипами
def render_canvas(img_b64, boxes, img_w, img_h, key):
    # Отрисовка изображения на canvas с тултипами для каждой рамки
    safe_key   = ''.join(c if c.isalnum() else '_' for c in str(key))
    boxes_json = json.dumps(boxes)
    html = f'''
    <canvas id="c{safe_key}" style="max-width:100%;cursor:crosshair"></canvas>
    <script>
    const canvas = document.getElementById('c{safe_key}');
    const ctx    = canvas.getContext('2d');
    const boxes  = {boxes_json};
    const img    = new Image();
    img.src = 'data:image/png;base64,{img_b64}';
    img.onload = () => {{ canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight; drawAll(null); }};
    function drawAll(hover) {{
      ctx.clearRect(0,0,canvas.width,canvas.height);
      ctx.drawImage(img,0,0);
      boxes.forEach((b,i) => {{
        const hue=(i*47)%360;
        ctx.strokeStyle='hsl('+hue+',85%,55%)';
        ctx.lineWidth=2;
        ctx.strokeRect(b.x1,b.y1,b.x2-b.x1,b.y2-b.y1);
        const label=b.cls+' '+(b.conf*100).toFixed(0)+'%';
        ctx.font='bold 14px sans-serif';
        const tw=ctx.measureText(label).width;
        ctx.fillStyle='hsl('+hue+',85%,55%)';
        ctx.fillRect(b.x1,b.y1-20,tw+8,20);
        ctx.fillStyle='#fff';
        ctx.fillText(label,b.x1+4,b.y1-5);
        if(i===hover){{
          const lines=['X1,Y1: ('+b.x1+','+b.y1+')',
            'X2,Y2: ('+b.x2+','+b.y2+')','Center: ('+b.xc+','+b.yc+')'];
          const lh=20,pad=8;
          ctx.font='13px monospace';
          const tw2=Math.max(...lines.map(l=>ctx.measureText(l).width));
          const bw=tw2+pad*2,bh=lines.length*lh+pad*2;
          let tx=b.x2+6,ty=b.y1;
          if(tx+bw>canvas.width) tx=b.x1-bw-6;
          if(ty+bh>canvas.height) ty=canvas.height-bh-4;
          ctx.fillStyle='rgba(15,15,25,0.88)';
          ctx.beginPath();ctx.roundRect(tx,ty,bw,bh,6);ctx.fill();
          ctx.fillStyle='#e2e8f0';
          lines.forEach((line,li)=>{{
            ctx.fillText(line,tx+pad,ty+pad+(li+1)*lh-4);
          }});
        }}
      }});
    }}
    canvas.addEventListener('mousemove',e=>{{
      const r=canvas.getBoundingClientRect();
      const sx=canvas.width/r.width,sy=canvas.height/r.height;
      const mx=(e.clientX-r.left)*sx,my=(e.clientY-r.top)*sy;
      let hover=null;
      boxes.forEach((b,i)=>{{
        if(mx>=b.x1&&mx<=b.x2&&my>=b.y1&&my<=b.y2) hover=i;
      }});
      drawAll(hover);
    }});
    canvas.addEventListener('mouseleave',()=>drawAll(null));
    </script>'''
    components.html(html, height=int(700*img_h/img_w)+30)

# Статичная отрисовка для кнопки Download
def draw_static(img_array, boxes):
    out = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    for i,b in enumerate(boxes):
        hue=int(i*180/max(len(boxes),1))
        clr=tuple(int(x) for x in cv2.cvtColor(
              np.uint8([[[hue,220,220]]]),cv2.COLOR_HSV2BGR)[0][0])
        cv2.rectangle(out,(b['x1'],b['y1']),(b['x2'],b['y2']),clr,2)
        lbl=f"{b['cls']} {b['conf']:.0%}"
        (tw,th),bl=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.6,2)
        cv2.rectangle(out,(b['x1'],b['y1']-th-bl-4),
                         (b['x1']+tw,b['y1']),clr,-1)
        cv2.putText(out,lbl,(b['x1'],b['y1']-bl-1),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

# Модель кэшируется и не перезагружается при каждом взаимодействии
model       = load_model()
class_names = model.names

with st.sidebar:
    st.title('Settings')
    confidence=st.slider('Confidence threshold',0.1,1.0,0.5,0.05)
    iou=st.slider('IoU threshold (NMS)',0.1,1.0,0.45,0.05)
    image_size=st.selectbox('Input size',[320,416,512,640,1024],index=3)
    st.divider()
    if st.checkbox('Filter classes'):
        selected=st.multiselect('Keep only:',
                    list(class_names.values()),
                    default=list(class_names.values()))
        class_filter=[k for k,v in class_names.items() if v in selected]
    else:
        class_filter=None
    st.caption(f'Model classes: {len(class_names)}')

st.title('YOLO Object Detection')

uploaded_files=st.file_uploader('Upload images',
    type=['jpg','jpeg','png','bmp','webp'],accept_multiple_files=True)

if not uploaded_files:
    st.info('Upload one or more images to get started')
    st.stop()

if not st.button('Run detection',type='primary',use_container_width=True):
    st.stop()

# Обработка каждого файла
for f in uploaded_files:
    st.divider()
    st.subheader(f.name)
    image=Image.open(f).convert('RGB')
    img_array=np.array(image)
    with st.spinner(f'Processing {f.name}...'):
        result=model.predict(source=img_array,conf=confidence,
            iou=iou,imgsz=image_size,classes=class_filter,verbose=False)[0]
    n=len(result.boxes) if result.boxes else 0
    boxes_data=[]
    for box in (result.boxes or []):
        cls_id=int(box.cls[0]); conf_val=float(box.conf[0])
        x1,y1,x2,y2=map(int,box.xyxy[0])
        boxes_data.append({'cls':class_names[cls_id],'conf':round(conf_val,3),
            'x1':x1,'y1':y1,'x2':x2,'y2':y2,'xc':(x1+x2)//2,'yc':(y1+y2)//2})
    col1,col2=st.columns(2)
    with col1:
        st.caption('Original')
        st.image(image,use_container_width=True)
    with col2:
        st.caption(f'Result – {n} object(s). Hover over boxes.')
        buf=BytesIO(); image.save(buf,format="PNG")
        render_canvas(base64.b64encode(buf.getvalue()).decode(),
                      boxes_data,image.width,image.height,key=f.name)
        if n>0:
            buf2=BytesIO()
            Image.fromarray(draw_static(img_array,boxes_data)).save(buf2,"PNG")
            st.download_button('Download result',data=buf2.getvalue(),
                file_name=f'result_{f.name}',mime='image/png',key=f'dl_{f.name}')
    if n>0:
        st.dataframe(pd.DataFrame([{'Class':b['cls'],
            'Confidence':f"{b['conf']:.1%}",'X1,Y1':f"({b['x1']},{b['y1']})",
            'X2,Y2':f"({b['x2']},{b['y2']})",
            'Center':f"({b['xc']},{b['yc']})"}
            for b in boxes_data]),use_container_width=True,hide_index=True)
    else:
        st.warning('No objects found. Try lowering the confidence threshold.')
