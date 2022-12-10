import { useEffect, useState } from "react"
import DataTable from 'react-data-table-component';
import axios from "axios"

export default function Info() {
    const [dogs, setDogs] = useState([])
    const columns = [
        {
            name: 'Dogs',
            selector: row => row.dog,
        }
    ];

    useEffect(()=> {
        axios.get((process.env.NEXT_PUBLIC_HOST ? `${process.env.NEXT_PUBLIC_HOST}` : '') + '/api/dogs/')
      .then(({ data }) => {
        setDogs(data);
        }
      );
    }, [])

    const customStyles = {
      
        headCells: {
            style: {
                display: 'flex',
                justifyContent: 'center',
                fontSize: '22px'
                
            },
        },
        cells: {
            style: {
                paddingLeft: '16px',
                fontSize: '18px'
            },
        },
    };

    return (
           <div className="pt-5 w-50 mx-auto">
             <DataTable
                className="border border-gray border-bottom-0"
                columns={columns}
                data={dogs.map((dog, index)=>{
                    return  {id: index, dog}
                })}
                pagination
                customStyles={customStyles}
                
            />
           </div>
    )
}