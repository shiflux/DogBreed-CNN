import { useState, useRef } from 'react';
import axios from 'axios'

export default function Home() {
  const [fileImage, setFileImage] = useState(null);
  const [responseData, setResponseData] = useState(null);
  const [requestInProgress, setRequestInProgress] = useState(false);
  const inputRef = useRef(null);

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
    inputRef.current.value = null;
    }
  )
  .catch((error) => {
    setRequestInProgress(false);
    setFileImage(null);
    inputRef.current.value = null;
  });
  }

  return (
    <div>

      
      <div className="p-5 mt-5 container text-center bg-image rounded-3" style={{backgroundImage: "url('/hero.jpg')",height: '300px', backgroundSize: 'cover', backgroundRepeat:'no-repeat', backgroundPosition:'center'}} >
        <div className="mask p-4" style={{backgroundColor: 'rgba(0, 0, 0, 0.4)'}}>
          <div className="d-flex justify-content-center align-items-center h-100">
            <div className="text-white">
              <h1 className="mb-3">Dog Breed Finder</h1>
              <h4 className="mb-3">Upload an image of a dog and find its breed! <br/> You can also upload an image of a person and find the dog that looks the most like them.</h4>
              {/* <a className="btn btn-outline-light btn-lg" href="#!" role="button">Call to action</a> */}
            </div>
          </div>
        </div>
      </div>

      <div className='p-5 d-flex justify-content-center'>
        <input ref={inputRef} disabled={requestInProgress} type='file' onChange={(event) => {
          setFileImage(event.target.files[0]);
          setResponseData(null);
        }} />
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
           {responseData.image_type != 'unknown' &&
            <>{responseData.image_type == 'human' && <h4 class="card-title text-light">This human looks like a</h4>}
            <h4 class="card-title text-light">{responseData?.breed?.name}</h4>
            </>
            }
            {responseData.error && <h4 class="card-title text-light">Invalid file</h4>}
            {responseData.image_type == 'unknown' &&
            <h4 class="card-title text-light">Subject not recognized</h4>
            }
            <p class="card-text"></p>
          </div>
        </div>}

    </div>
  )
}