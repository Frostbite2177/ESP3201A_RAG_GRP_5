import json, subprocess, sys, threading, tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
from benchmark_oneclick import questions
ROOT=Path(__file__).resolve().parent
class GUI(tk.Tk):
 def __init__(self):
  super().__init__();self.title('PyPSA RAG Benchmark - One Click');self.geometry('900x720')
  self.repo=tk.StringVar(value=str(ROOT/'PyPSA-master'));self.qfile=tk.StringVar(value=str(ROOT/'output/hifirepoqa/questions.json'))
  self.start=tk.StringVar(value='0');self.count=tk.StringVar(value='1');self.indices=tk.StringVar();self.llm=tk.StringVar(value='qwen2.5-coder:3b');self.topk=tk.StringVar(value='3');self.rebuild=tk.BooleanVar(value=False)
  f=ttk.Frame(self,padding=15);f.pack(fill='both',expand=True)
  ttk.Label(f,text='PyPSA: Baseline / AST / Hybrid / Parent',font=('Segoe UI',16,'bold')).pack(anchor='w')
  for label,var,browse in [('PyPSA-master folder',self.repo,True),('Questions JSON file',self.qfile,False),('Ollama model',self.llm,None),('Start question index (0-based)',self.start,None),('Number of questions',self.count,None),('Specific indices (optional, comma-separated)',self.indices,None),('Top K retrieved chunks',self.topk,None)]:
   row=ttk.Frame(f);row.pack(fill='x',pady=4);ttk.Label(row,text=label,width=39).pack(side='left');ttk.Entry(row,textvariable=var).pack(side='left',fill='x',expand=True)
   if browse is not None:ttk.Button(row,text='Browse',command=lambda v=var,isdir=browse:v.set(filedialog.askdirectory() if isdir else filedialog.askopenfilename(filetypes=[('JSON','*.json')]))).pack(side='left')
  ttk.Checkbutton(f,text='Rebuild all indexes even if verified indexes exist',variable=self.rebuild).pack(anchor='w',pady=5)
  buttons=ttk.Frame(f);buttons.pack(fill='x',pady=10)
  ttk.Button(buttons,text='Show available questions',command=self.showq).pack(side='left',padx=4)
  self.runbtn=ttk.Button(buttons,text='RUN ALL 4 ARCHITECTURES',command=self.launch);self.runbtn.pack(side='left',padx=4)
  ttk.Button(buttons,text='Open results folder',command=self.open_results).pack(side='left',padx=4)
  self.info=ttk.Label(f,text='Question indices start at 0. Only questions present in the selected JSON can be run.');self.info.pack(anchor='w')
  self.log=tk.Text(f,wrap='word',font=('Consolas',9));self.log.pack(fill='both',expand=True,pady=8)
  self.showq()
 def write(self,s):self.log.insert('end',s);self.log.see('end')
 def showq(self):
  try:
   rows=questions(self.qfile.get());self.info.config(text=f'{len(rows)} usable Q&A entries found. Indices: 0 to {len(rows)-1}.')
   self.write('\nAVAILABLE QUESTIONS:\n'+''.join(f'[{i}] ID {r["id"]}: {r["question"].splitlines()[0][:100]}\n' for i,r in enumerate(rows))+'\n')
  except Exception as e:self.info.config(text=f'Cannot load questions: {e}')
 def open_results(self):
  import os
  folder=ROOT/'output/benchmark_runs';folder.mkdir(parents=True,exist_ok=True)
  if sys.platform=='win32':os.startfile(str(folder))
  else:messagebox.showinfo('Results',str(folder))
 def launch(self):
  try:
   if not Path(self.repo.get()).is_dir():raise ValueError('Select an extracted PyPSA-master folder')
   rows=questions(self.qfile.get());start=int(self.start.get());count=int(self.count.get());k=int(self.topk.get())
   indices=[int(x.strip()) for x in self.indices.get().split(',') if x.strip()] if self.indices.get().strip() else list(range(start,start+count))
   if not indices or min(indices)<0 or max(indices)>=len(rows) or count<1 or k<1:raise ValueError(f'Indices must be within 0..{len(rows)-1}, count and top K positive')
  except Exception as e:messagebox.showerror('Invalid configuration',str(e));return
  cmd=[sys.executable,'-u',str(ROOT/'benchmark_oneclick.py'),'--repo',self.repo.get(),'--questions',self.qfile.get(),'--start',str(start),'--count',str(count),'--top-k',str(k),'--llm',self.llm.get()]
  if self.indices.get().strip():cmd+=['--indices',self.indices.get()]
  if self.rebuild.get():cmd+=['--rebuild']
  self.runbtn.config(state='disabled');self.write('\nSTARTING: '+ ' '.join(cmd)+'\n')
  def work():
   try:
    p=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1)
    for line in p.stdout:self.after(0,self.write,line)
    code=p.wait();self.after(0,self.write,f'\nProcess finished with exit code {code}\n')
   except Exception as e:self.after(0,self.write,'ERROR: '+str(e)+'\n')
   finally:self.after(0,lambda:self.runbtn.config(state='normal'))
  threading.Thread(target=work,daemon=True).start()
if __name__=='__main__':GUI().mainloop()
