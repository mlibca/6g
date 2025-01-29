#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NAME_LENGTH 100

typedef struct {
    char name[MAX_NAME_LENGTH];
    int age;
} Person;

void print_person_details(Person* p) {
    if (p != NULL) {
        printf("Name: %s, Age: %d\n", p->name, p->age);
    }
    else {
        printf("Invalid person object.\n");
    }
}

void read_person_from_file(const char* filename, Person* p) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return;
    }

    if (fscanf(file, "%s %d", p->name, &p->age) != 2) {
        fprintf(stderr, "Error reading data from file.\n");
    }

    fclose(file);
}

int main() {
    Person person;

    printf("Enter the name: ");
    fgets(person.name, MAX_NAME_LENGTH, stdin);
    person.name[strcspn(person.name, "\n")] = '\0'; // Remove newline from fgets

    printf("Enter the age: ");
    scanf("%d", &person.age);

    print_person_details(&person);

    // Save person details to file
    FILE* file = fopen("person.txt", "w");
    if (file != NULL) {
        fprintf(file, "%s %d\n", person.name, person.age);
        fclose(file);
    }
    else {
        perror("Error opening file for writing");
    }

    // Reading from file
    Person file_person;
    read_person_from_file("person.txt", &file_person);
    print_person_details(&file_person);

    return 0;
}
