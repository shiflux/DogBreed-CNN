import { useState } from 'react';
import axios from 'axios'

export default function Home() {
  const [fileImage, setFileImage] = useState(null)
  const [responseData, setResponseData] = useState(null)
  const [requestInProgress, setRequestInProgress] = useState(false)

  const handleSubmission = (file) => {
    const formData = new FormData();
    formData.append('file', file);
    setRequestInProgress(true);
    setResponseData(null);
    axios.post((process.env.NEXT_PUBLIC_HOST ? `${process.env.NEXT_PUBLIC_HOST}` : '') + '/api/predict_dog/',
    formData, {
      headers: {
        'Content-Type': 'binary'
      }
  })
  .then(({ data }) => {
    setResponseData(data);
    setRequestInProgress(false);
    setFileImage(null);
    }
  )
  .catch((error) => {
    setRequestInProgress(false);
    setFileImage(null);
  });
  }

  return (
    <div>


      <div className='p-5 d-flex justify-content-center'> 
        <input disabled={requestInProgress} type='file' onChange={(event) => {setFileImage(event.target.files[0]); setResponseData(null);}} />
        <button disabled={fileImage==null} onClick={() => {handleSubmission(fileImage)}} className="btn btn-dark" style={{width: 'fit-content'}}>
        Identify
        </button>
      </div>

      {requestInProgress &&
      
      <div className="d-flex justify-content-center flex-column">
        <div className="spinner-grow mx-auto bg-transparent" role="status">
          <span className="sr-only"><img src='/logo.png' width={40}></img></span>
          
          </div>
        <p className='mx-auto pt-3'>Predicting...</p>
      </div>
      }

      {responseData &&  <div class="card bg-dark mx-auto mt-5 text-center" style={{minWidth: '320px', maxWidth:'500px'}}>
          <div class="card-body">
            <h4 class="card-title text-light">{responseData?.breed?.name}</h4>
            <p class="card-text"></p>
          </div>
        </div>}

    </div>
  )
}