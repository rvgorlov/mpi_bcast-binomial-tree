#include <iostream>
#include <cstdlib>
#include <string>

#include "/usr/include/mpi/mpi.h"

#define N 16 * 1024 * 1024

// Функция Bcast основанная на биноминальных деревьях
static void MPI_Bcast_binominal_three(
        void *buffer,               // Буффер для передачи
        int count,                  // Размер массива
        MPI_Datatype datatype,
        int root,
        MPI_Comm comm)
{
    int size, rank, src, dst;

    MPI_Status status;

    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    if(size == 1) return;

    int relative_rank = (rank >= root) ? rank - root : rank - root + size;

    int mask = 0x1;

    while (mask < size)
    {
        if (relative_rank & mask)
        {
            src = rank - mask;

            if (src < 0)
                src += size;

            PMPI_Recv(buffer, count, datatype, src, 0, comm, &status);

            break;
        }
        mask <<= 1;
    }


    mask >>= 1;

    while (mask > 0)
    {
        if (relative_rank + mask < size)
        {
            dst = rank + mask;

            if (dst >= size)
            {
                dst -= size;
            }

            PMPI_Send(buffer, count, datatype, dst, 0, comm);
            // std::cout << rank << ">>" << dst << std::endl;
        }
        mask >>= 1;
    }
}

int main(int argc, char **argv)
{
    int rank, size, root = 0;
    int buf = 0;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        buf = 99999;
        std::cout << "==========Binomial_three Bcast==========" << std::endl;
        std::cout << "Size = " << size << std::endl;
    }

    for (int i = 0; i < size; ++i)
    {
        if(i == rank)
        {
            std::cout << "Rank = " << rank << " before buf = " << buf << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //MPI_Bcast(&buf, 1 ,MPI_INT, root, MPI_COMM_WORLD);
    MPI_Bcast_binominal_three(&buf, 1 ,MPI_INT, root, MPI_COMM_WORLD);

    for (int i = 0; i < size; ++i)
    {
        if(i == rank)
        {
            std::cout << "Rank = " << rank << " before buf = " << buf << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
