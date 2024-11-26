/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */



#ifndef DATALOADER_H
#define DATALOADER_H
#include "ann/xtensor_lib.h"
#include "ann/dataset.h"
#include "list/listheader.h"



using namespace std;



template<typename DType, typename LType>
class DataLoader {
public:
    class Iterator; // forward declaration
private:
    Dataset<DType, LType>* pointer;
    int batch_size;
    bool shuffle;
    bool drop_last;
    int m_seed;

    int num_batches;
    xt::xarray<int> idx_arr;
    XArrayList<Batch<DType, LType>> batch;
    
public:
    DataLoader(Dataset<DType, LType>* pointer,int batch_size,bool shuffle=true,bool drop_last=false, int seed=-1) {
        if (!pointer) return;
        idx_arr = xt::arange<unsigned long>(pointer->len());
        this->shuffle = shuffle;
        this->drop_last = drop_last; 
        this->m_seed = seed;

        // if ((this->shuffle==true)){
        //     xt::random::default_engine_type engine(0);
        //     xt::random::seed(this->m_seed);
        //     xt::random::shuffle(idx_arr,engine);
        // }
        if ((this->shuffle==true)){
            if ((this->m_seed >= 0)){
                xt::random::seed(this->m_seed);
                xt::random::shuffle(idx_arr);
            }
            else{
                xt::random::shuffle(idx_arr);
            }
        }

        this->pointer = pointer;
        this->batch_size = batch_size;
               
        int dataset_len = pointer->len();
        int full_batches = pointer->len() / batch_size;
        int remaining_samples = pointer->len() % batch_size;
        if (dataset_len<=0) return;

        num_batches = full_batches;
        if ( (dataset_len > 0) && (dataset_len < batch_size) && (drop_last==false) ) num_batches = 0; //Special case 

        this->batch = xvector<Batch<DType, LType>>(nullptr,nullptr,num_batches);
        TensorDataset<DType, LType>* tensor_ds = dynamic_cast<TensorDataset<DType, LType>*>(pointer);
        if (!tensor_ds) return;
        xt::xarray<DType> all_data = tensor_ds->get_data_slice(0, dataset_len);
        xt::xarray<LType> all_labels;
        bool has_labels = !tensor_ds->get_label_shape().empty();  

        
        if (has_labels) {
            all_labels = tensor_ds->get_label_slice(0, dataset_len);
        }

        
        xt::xarray<DType> shuffled_data = xt::view(all_data, xt::keep(idx_arr));
        xt::xarray<LType> shuffled_labels;
        if (has_labels) {
            shuffled_labels = xt::view(all_labels, xt::keep(idx_arr));
        }

        // Batching
        for (int i = 0; i < num_batches; i++) {
            int start_idx = i * batch_size;
            int end_idx = std::min(start_idx + batch_size, dataset_len);

            xt::xarray<DType> batch_data = xt::view(shuffled_data, xt::range(start_idx, end_idx));

            if (has_labels) {
                xt::xarray<LType> batch_label = xt::view(shuffled_labels, xt::range(start_idx, end_idx));
                batch.add(Batch<DType, LType>(batch_data, batch_label));
            } else {
                
                batch.add(Batch<DType, LType>(batch_data, xt::xarray<LType>()));  
            }
        }

        // Handling remaining samples if drop_last is false
        if ((remaining_samples > 0) && (drop_last == false) && (dataset_len > batch_size)) {
            int start_idx = full_batches * batch_size;
            int end_idx = dataset_len;

            Batch<DType, LType>& last_batch = batch.get(batch.size() - 1);
            xt::xarray<DType> remaining_data = xt::view(shuffled_data, xt::range(start_idx, end_idx));

            if (has_labels) {
                xt::xarray<LType> remaining_labels = xt::view(shuffled_labels, xt::range(start_idx, end_idx));
                xt::xarray<DType> new_data = xt::concatenate(xt::xtuple(last_batch.getData(), remaining_data), 0);
                xt::xarray<LType> new_labels = xt::concatenate(xt::xtuple(last_batch.getLabel(), remaining_labels), 0);

                last_batch.setData(new_data);
                last_batch.setLabel(new_labels);
            } else {
                xt::xarray<DType> new_data = xt::concatenate(xt::xtuple(last_batch.getData(), remaining_data), 0);
                last_batch.setData(new_data);
            }
        }
    }
    virtual ~DataLoader() {}
    

    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// START: Section                                                     //
    /////////////////////////////////////////////////////////////////////////

    Iterator begin() {
        return Iterator(this, 0);
    }

    Iterator end() {
        return Iterator(this, num_batches);
    }

    class Iterator {
    private:
        DataLoader<DType,LType> *loader;
        size_t curr_batch_idx;
        
    public:
        Iterator(DataLoader<DType,LType> *loader, size_t curr_batch_idx){
            this->loader = loader;
            this->curr_batch_idx = curr_batch_idx;
        }

        Batch<DType, LType> &operator*() const{
            return loader->batch.get(curr_batch_idx);
        }

        

        Iterator& operator++() {
            ++curr_batch_idx;
            return *this;
        }

        Iterator operator++(int) {
            Iterator temp = *this;
            ++*this;
            return temp;
        }

        bool operator!=(const Iterator &other )const{
            return curr_batch_idx!=other.curr_batch_idx;
        }
        Iterator& operator=(const Iterator& it) {
            if (this != &it) {
                this->loader = it.loader;
                this->curr_batch_idx = it.curr_batch_idx;
            }
            return *this;
        }
        bool operator==(const Iterator &other )const{
            return (curr_batch_idx == other.curr_batch_idx);
        }
    };

    

    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// END: Section                                                       //
    /////////////////////////////////////////////////////////////////////////
};


#endif /* DATALOADER_H */

