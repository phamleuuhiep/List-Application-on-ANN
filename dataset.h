/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */



#ifndef DATASET_H
#define DATASET_H
#include "ann/xtensor_lib.h"
using namespace std;

template<typename DType, typename LType>
class DataLabel{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
public:
    DataLabel(xt::xarray<DType> data,  xt::xarray<LType> label):
    data(data), label(label){
    }
    xt::xarray<DType> getData() const{ return data; }
    xt::xarray<LType> getLabel() const{ return label; }
};

template<typename DType, typename LType>
class Batch{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
public:
    Batch(){}
    Batch(xt::xarray<DType> data,  xt::xarray<LType> label):
    data(data), label(label){
    }
    virtual ~Batch(){}
    xt::xarray<DType>& getData(){return data; }
    xt::xarray<LType>& getLabel(){return label; }
    void setData(const xt::xarray<DType>& data) {
        this->data = data;
    }
    void setLabel(const xt::xarray<LType>& label) {
        this->label = label;
    }
    bool operator==(Batch<DType,LType>& other){
        return this->getData() == other.getData() && this->getLabel() == other.getLabel();
    }
};
template<typename DType,typename LType>
std::ostream& operator<<(std::ostream& os, Batch<DType,LType>& b){
        // os << shape2str(b.getData().shape()) << endl;
        // os << shape2str(b.getLabel().shape()) << endl;
        os << " ";
        return os;
}



template<typename DType, typename LType>
class Dataset{
private:
public:
    Dataset(){};
    virtual ~Dataset(){};
    
    virtual int len()=0;
    virtual DataLabel<DType, LType> getitem(int index)=0;
    virtual xt::svector<unsigned long> get_data_shape()=0;
    virtual xt::svector<unsigned long> get_label_shape()=0;
    
};

//////////////////////////////////////////////////////////////////////
template<typename DType, typename LType>
class TensorDataset: public Dataset<DType, LType>{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
    xt::svector<unsigned long> data_shape, label_shape;
    
public:
    /* TensorDataset: 
     * need to initialize:
     * 1. data, label;
     * 2. data_shape, label_shape
    */
    TensorDataset(xt::xarray<DType> data, xt::xarray<LType> label){
        /* TODO: your code is here for the initialization
         */
        this->data = data;
        this->label = label;
        this->data_shape = xt::svector<unsigned long>(data.shape().begin(), data.shape().end());
        this->label_shape = xt::svector<unsigned long>(label.shape().begin(), label.shape().end());
    
    }
    /* len():
     *  return the size of dimension 0
    */
    int len(){
        /* TODO: your code is here to return the dataset's length
         */
        return this->data.shape(0);
    }
    
    /* getitem:
     * return the data item (of type: DataLabel) that is specified by index
     */
    DataLabel<DType, LType> getitem(int index){
        /* TODO: your code is here
         */
        if (index<0 || index>=this->len()) throw std::out_of_range("Index is out of range!");
        xt::xarray<DType> data_item;
        xt::xarray<LType> label_item;  
        bool do_not_has_data = (this->get_data_shape().size() == 0);
        bool do_not_has_label = (this->get_label_shape().size() == 0);
        data_item = ( do_not_has_data ? this->data : xt::view(this->data, index,xt::all()));
        label_item = ( do_not_has_label ? this->label : xt::view(this->label, index,xt::all()));
        return DataLabel<DType, LType>(data_item, label_item);
    }
    
    xt::svector<unsigned long> get_data_shape(){
        /* TODO: your code is here to return data_shape
         */
        return this->data_shape;
    }
    xt::svector<unsigned long> get_label_shape(){
        /* TODO: your code is here to return label_shape
         */
        return this->label_shape;
    }
    xt::xarray<LType> get_label_slice(int start_idx, int end_idx) {
        return xt::view(this->label, xt::range(start_idx, end_idx));
    }

    xt::xarray<DType> get_data_slice(int start_idx, int end_idx) {
        return xt::view(this->data, xt::range(start_idx, end_idx));
    }
};



#endif /* DATASET_H */

