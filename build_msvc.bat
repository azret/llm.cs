  @echo off 

  setlocal

  set obj=obj\x64\Debug\CUDA\dev\cuda
  IF NOT EXISTS "%obj%" (
	mkdir %obj%
  )
  rem nvcc --keep --keep-dir %obj% -objtemp -O3 --use_fast_math -Xcompiler dev\cuda\vectorAdd_kernel64.cu -o %obj%\vectorAdd_kernel64 -lcublas -lcublasLt
  nvcc --keep --keep-dir %obj% -O3 --use_fast_math -objtemp -Xcompiler dev\cuda\vectorAdd_kernel64.cu -o .%obj%\vectorAdd_kernel64
  del %obj% /f /s /q
  
  endlocal