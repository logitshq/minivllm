for(uint i=0;i<5;i++) descriptor[i]=0; descriptor[0]=value_ndim; for(uint i=0;i<value_ndim;i++){descriptor[1+i]=value_shape[i];descriptor[3+i]=value_strides[i];}
