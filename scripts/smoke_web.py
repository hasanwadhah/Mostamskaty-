"""Optional real-browser smoke check; start Streamlit before running this script."""
from pathlib import Path
import os
import shutil
import cv2
import numpy as np
from playwright.sync_api import sync_playwright, expect
from pypdf import PdfReader

root=Path(__file__).resolve().parents[1]
(root/'artifacts').mkdir(exist_ok=True)
scene=np.full((650,900,3),(39,48,58),np.uint8)
cv2.fillConvexPoly(scene,np.array([[140,130],[750,80],[800,480],[100,510]],np.int32),(220,235,220))
cv2.imwrite(str(root/'artifacts/web-input.png'),scene)
errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(executable_path=os.environ.get('CHROMIUM_EXECUTABLE') or shutil.which('chromium') or p.chromium.executable_path,headless=True,args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':1100},accept_downloads=True)
 page.on('pageerror',lambda e:errors.append(str(e)))
 page.goto(os.environ.get('MOSTAMSKATY_URL','http://127.0.0.1:8501'))
 page.locator('input[type=file]').first.wait_for(state='attached',timeout=30000)
 page.locator('input[type=file]').first.set_input_files(root/'artifacts/web-input.png')
 page.get_by_role('heading',name='تحرير البطاقة 1 — الوجه',exact=True).wait_for(timeout=30000)
 canvas=page.frame_locator('iframe').first.locator('#canvas')
 canvas.wait_for()
 frame=next(frame for frame in page.frames if frame!=page.main_frame and 'component' in frame.url)
 frame.wait_for_function('image !== null && points.length===4')
 old=frame.evaluate('({revision,x:points[0][0]*scale+offsetX,y:points[0][1]*scale+offsetY})')
 box=canvas.bounding_box()
 page.mouse.move(box['x']+old['x'],box['y']+old['y']);page.mouse.down()
 page.mouse.move(box['x']+old['x']+12,box['y']+old['y']+8,steps=6);page.mouse.up()
 page.get_by_role('button',name='تطبيق القص',exact=True).click()
 frame.wait_for_function('(old)=>revision !== old',arg=old['revision'])
 page.screenshot(path=str(root/'artifacts/web-editor.png'),full_page=True)
 page.get_by_role('button',name='اعتماد وتثبيت',exact=True).click()
 page.get_by_role('heading',name='تحرير البطاقة 1 — الوجه',exact=True).wait_for(state='hidden')
 expect(page.locator('input[type=file]').nth(1)).to_be_enabled(timeout=30000)
 page.locator('input[type=file]').nth(1).set_input_files(root/'artifacts/web-input.png')
 try:
  page.get_by_role('heading',name='تحرير البطاقة 1 — الظهر',exact=True).wait_for(timeout=12000)
 except Exception:
  print('HEADINGS:',page.locator('h1,h2,h3').all_text_contents())
  print('FILE INPUTS:',page.locator('input[type=file]').evaluate_all('(els)=>els.map(x=>({label:x.getAttribute("aria-label"),disabled:x.disabled,files:[...x.files].map(f=>f.name)}))'))
  print('ALERTS:',page.locator('[data-testid=stAlert]').all_text_contents())
  page.screenshot(path=str(root/'artifacts/web-debug.png'),full_page=True)
  raise
 page.get_by_role('button',name='اعتماد وتثبيت',exact=True).click()
 page.get_by_role('heading',name='تحرير البطاقة 1 — الظهر',exact=True).wait_for(state='hidden')
 page.get_by_role('button',name='تجهيز ملف التنزيل',exact=True).click()
 print('Both sides accepted; export requested.',flush=True)
 download_button=page.get_by_text('تنزيل الملف الجاهز',exact=True)
 try:
  download_button.wait_for(timeout=15000)
 except Exception:
  print('EXPORT ALERTS:',page.locator('[data-testid=stAlert]').all_text_contents())
  print('BUTTONS:',page.locator('button,a').all_text_contents())
  page.screenshot(path=str(root/'artifacts/web-export-debug.png'),full_page=True)
  raise
 with page.expect_download() as download:
  download_button.click()
 destination=str(root/'artifacts/web-smoke-result.pdf')
 download.value.save_as(destination)
 pdf=PdfReader(destination)
 assert len(pdf.pages)==1 and len(pdf.pages[0].images)==2
 page.screenshot(path=str(root/'artifacts/web-ready.png'),full_page=True)
 page.get_by_role('button',name='مسح الجلسة بالكامل',exact=True).click()
 page.get_by_role('button',name='تجهيز ملف التنزيل',exact=True).wait_for()
 expect(page.get_by_role('button',name='تجهيز ملف التنزيل',exact=True)).to_be_disabled(timeout=15000)
 assert not errors,errors
 print('Browser workflow passed: upload, drag crop corner, perspective crop, front/back acceptance, A4 PDF download, and session clearing. No JavaScript errors.')
 browser.close()
